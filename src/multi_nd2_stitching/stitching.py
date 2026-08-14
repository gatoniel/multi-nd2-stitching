import nd2
import zarr
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from tqdm import trange, tqdm
import scipy.fft as spfft
import os
from pathlib import Path
from itertools import combinations
from .config import load_config, get_slices


def get_position_names(nd2_file):
    return [pos.name for pos in nd2_file.experiment[1].parameters.points]


def get_position(position_names, names):
    pos = None
    for i, name in enumerate(position_names):
        if name in names:
            if pos is None:
                pos = i
            else:
                raise ValueError("non-unique position_names")
    if pos is None:
        raise ValueError("position name not found", position_names)
    return pos


def open_handle_pool(path: str, n_handles: int = 10) -> Queue:
    """Öffnet n_handles ND2File-Objekte und legt sie in eine Queue."""
    pool: Queue = Queue()
    for _ in range(n_handles):
        pool.put(nd2.ND2File(path))
    return pool


def close_handle_pool(pool: Queue) -> None:
    while not pool.empty():
        pool.get().close()


def read_frame_from_pool(pool: Queue, frame_index: int) -> float:
    """Holt ein freies Handle aus dem Pool, liest einen Frame, gibt es zurück."""
    f = pool.get()  # blockiert, bis ein Handle frei ist
    try:
        arr = f.read_frame(frame_index)
    finally:
        pool.put(f)
    return arr


def _phase_corr_from_ffts(fft0, fft1, shape, workers=-1):
    mult = fft0 * np.conjugate(fft1)
    mult /= np.abs(mult)
    inverse = spfft.irfftn(mult, s=shape, workers=workers, axes=list(range(fft0.ndim)))
    inverse_shifted = np.fft.fftshift(inverse)
    peak = np.array(np.unravel_index(np.argmax(inverse_shifted), shape=shape))
    return peak - np.array(shape) // 2


def fft_translation_3d(img0, img1, workers=-1):
    fft0 = spfft.rfftn(img0, workers=workers)
    fft1 = spfft.rfftn(img1, workers=workers)
    return _phase_corr_from_ffts(fft0, fft1, img0.shape, workers=workers)


def crop_for_alignment(frame_parent, frame_child, axis, shift_px):
    """Cheap numpy-slice crop, applied AFTER the z-restricted frame is
    already in memory (no extra disk I/O here)."""
    n = frame_parent.shape[axis]
    sl_parent = [slice(None)] * 3
    sl_child = [slice(None)] * 3

    sl_parent[axis] = slice(shift_px, None)
    sl_child[axis] = slice(None, n - shift_px)

    return frame_parent[tuple(sl_parent)], frame_child[tuple(sl_child)]


class PositionAlignment:
    def __init__(self, file_path):
        self.config = load_config(file_path)
        self.file_path = file_path
        self.folder = file_path.replace(".yaml", "")
        Path(self.folder).mkdir(parents=True, exist_ok=True)
        self.path_offsets_neighbors = os.path.join(self.folder, "offsets_neighbors.npy")
        self.path_offsets_time = os.path.join(self.folder, "offsets_time.npy")
        self.path_canvas = os.path.join(self.folder, "stitched.zarr")

        self.nd2_files = tuple(nd2.ND2File(file) for file in self.config.files)
        self._init_sizes()
        self._init_names_positions()
        self._init_neighborhood()
        self._init_imgs()

        self.loaded = False

        self.start_names_dict = self.config.start_names
        self.start_names = list(self.start_names_dict)
        self.start_names_by_file = {i: [] for i in range(len(self.config.files))}
        for name in self.start_names:
            for file_number in self.start_names_dict[name]:
                self.start_names_by_file[file_number].append(name)

    def _init_sizes(self):
        self.nzs = tuple(f.sizes["Z"] for f in self.nd2_files)

        self.nxs = tuple(f.sizes["X"] for f in self.nd2_files)
        assert all(self.nxs[0] == x for x in self.nxs[1:])

        self.nys = tuple(f.sizes["Y"] for f in self.nd2_files)
        assert all(self.nys[0] == y for y in self.nys[1:])

        self.nx = self.nxs[0]
        self.ny = self.nys[0]

        self.max_nz = max(self.nzs)
        self.min_nz = min(self.nzs)
        self.nts = tuple(
            f.sizes["T"] if ("T" in f.sizes) else 1 for f in self.nd2_files
        )
        self.max_t_by_file = np.cumsum(self.nts)
        self.nt = sum(self.nts)

    def _init_names_positions(self):
        self.position_names = tuple(get_position_names(f) for f in self.nd2_files)

        self.names = {}
        self.starts = {}
        self.ends = {}
        self.positions = {}

        for name, pos in self.config.names.items():
            tmp_names = [name]
            try:
                for name2 in pos.aliases:
                    tmp_names.append(name2)
            except TypeError:
                pass
            self.names[name] = tuple(tmp_names)
            self.starts[name] = tuple(pos.start)

            if pos.end is None:
                self.ends[name] = len(self.nd2_files)
            else:
                self.ends[name] = pos.end

            self.positions[name] = tuple(
                get_position(n, tmp_names)
                if self.ends[name] > i >= pos.start[0]
                else None
                for i, n in enumerate(self.position_names)
            )
        self.names_list = list(self.names.keys())

    def _init_neighborhood(self):
        neighbors_x = []
        neighbors_y = []

        low = self.config.grid_spacing - self.config.grid_spacing_error
        high = self.config.grid_spacing + self.config.grid_spacing_error

        for i in range(len(self.nd2_files)):
            file = self.nd2_files[i]
            tmp_positions = {
                pos[i]: name
                for name, pos in self.positions.items()
                if pos[i] is not None
            }
            stages = {
                name: file.experiment[1].parameters.points[i].stagePositionUm
                for i, name in tmp_positions.items()
            }
            um_positions = {
                name: np.array((pos.x, pos.y)) for name, pos in stages.items()
            }

            diffs = {}
            for i, j in combinations(um_positions.keys(), 2):
                diffs[(i, j)] = um_positions[i] - um_positions[j]

            for (name_i, name_j), diff in diffs.items():
                dx, dy = diff

                if np.abs(dx) < self.config.grid_spacing_error:
                    if low < dy < high:
                        neighbors_y.append((name_i, name_j))
                    if -high < dy < -low:
                        neighbors_y.append((name_j, name_i))
                if np.abs(dy) < self.config.grid_spacing_error:
                    if low < dx < high:
                        neighbors_x.append((name_i, name_j))
                    if -high < dx < -low:
                        neighbors_x.append((name_j, name_i))

        self.neighbors_x = list(set(neighbors_x))
        self.neighbors_y = list(set(neighbors_y))

        if self.config.flip_x:
            self.neighbors_x = [p[::-1] for p in self.neighbors_x]
        if self.config.flip_y:
            self.neighbors_y = [p[::-1] for p in self.neighbors_y]

        # self.neighbors_x_inds = [
        #     [(self.positions[i][k], self.positions[j][k]) for i, j in self.neighbors_x]
        #     for k in range(len(self.config.files))
        # ]
        # self.neighbors_y_inds = [
        #     [(self.positions[i][k], self.positions[j][k]) for i, j in self.neighbors_y]
        #     for k in range(len(self.config.files))
        # ]
        self.neighborhoods = sorted(
            [(i, j, 2) for i, j in self.neighbors_x]
            + [(i, j, 1) for i, j in self.neighbors_y]
        )

        if self.config.shift_px is None:
            self.shift_px = int(
                self.config.grid_spacing / self.nd2_files[0].voxel_size().x
            )
        else:
            self.shift_px = self.config.shift_px

    def _init_imgs(self):
        self.time_offsets = {}
        self.end_times = {}
        for name, inds in self.positions.items():
            self.time_offsets[name] = (
                sum(self.nts[: self.starts[name][0]]) + self.starts[name][1]
            )
            self.end_times[name] = sum(self.nts[: self.ends[name]])

        self.neighborhoods_start_time = []
        for name0, name1, _ in self.neighborhoods:
            off0 = self.time_offsets[name0]
            off1 = self.time_offsets[name1]
            self.neighborhoods_start_time.append(max(off0, off1))

        self.neighborhoods_end_time = []
        for name0, name1, _ in self.neighborhoods:
            end0 = self.end_times[name0]
            end1 = self.end_times[name1]
            self.neighborhoods_end_time.append(min(end0, end1))

    def __del__(self):
        for nd2_file in self.nd2_files:
            nd2_file.close()

    def read_vol(
        self,
        name,
        file_i,
        local_time,
        pool,
        executor,
    ):
        pos = self.positions[name][file_i]
        if pos is None:
            raise ValueError(f"pos ({pos}) not available in file ({file_i}).")
        file = self.nd2_files[file_i]
        try:
            if "T" in file.sizes:
                start = file._seq_index_from_coords((local_time, pos, 0))
            else:
                start = file._seq_index_from_coords((pos, 0))
        except TypeError as e:
            print(local_time, pos, 0, file, file.sizes)
            raise e

        arr = np.empty((self.min_nz, self.ny, self.nx), dtype=file.dtype)

        futures = [
            executor.submit(read_frame_from_pool, pool, i)
            for i in range(start, start + self.min_nz)
        ]
        for i, fut in enumerate(futures):
            arr[i] = fut.result()
        return arr

    def read_vol_simple(self, time, name):
        file_i = 0
        for t in self.max_t_by_file:
            if time < t:
                break
            else:
                file_i += 1
        else:
            raise ValueError(f"time ({time}) is out of range.")

        pos = self.positions[name][file_i]
        if pos is None:
            raise ValueError(f"pos ({pos}) not available at this time ({time}).")
        file = self.nd2_files[file_i]
        local_time = time - sum(self.nts[:file_i])
        if "T" in file.sizes:
            start = file._seq_index_from_coords((local_time, pos, 0))
        else:
            start = file._seq_index_from_coords((pos, 0))
        arr = np.empty((self.min_nz, self.ny, self.nx), dtype=file.dtype)
        for i in range(self.min_nz):
            arr[i] = file.read_frame(start + i)
        return arr

    def load(self):
        self.loaded = True
        self.offsets_neighbors = np.load(self.path_offsets_neighbors)
        self.offsets_time = np.load(self.path_offsets_time)

    def save(self):
        np.save(
            self.path_offsets_neighbors,
            self.offsets_neighbors,
        )
        np.save(self.path_offsets_time, self.offsets_time)

    def all_offsets(
        self,
        slices=None,
        workers_per_fft=-1,
        workers=10,
        t0=-1,
        t1=None,
    ):
        if slices is None:
            slices = get_slices(self.config.slices, self.min_nz)
        if t1 is None:
            t1 = self.nt

        n = len(self.start_names)

        if not self.loaded:
            self.offsets_time = np.zeros(
                (self.nt - 1, n, 3), dtype=int
            )  # offsets_time are given by time, start_name and x,y,z
            self.offsets_neighbors = np.zeros(
                (self.nt, len(self.neighborhoods), 3), dtype=int
            )

        prev_fft = {}
        current_start_time = 0
        for i, file in enumerate(self.config.files):
            tmp_start_names = self.start_names_by_file[i]
            pool = open_handle_pool(file, workers)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for t_ in trange(self.nts[i]):
                    t = t_ + current_start_time
                    if not t0 < t < t1:
                        continue
                    frames = {}
                    for start_name in tmp_start_names:
                        s_ind = self.start_names.index(start_name)
                        frame0 = self.read_vol(start_name, i, t_, pool, ex)
                        cur_fft = spfft.rfftn(frame0[slices], workers=workers_per_fft)
                        if start_name in prev_fft:
                            self.offsets_time[t - 1, s_ind] = _phase_corr_from_ffts(
                                prev_fft[start_name],
                                cur_fft,
                                frame0.shape,
                                workers=workers_per_fft,
                            )
                        prev_fft[start_name] = cur_fft
                        frames[start_name] = frame0

                    for k, start_time in enumerate(self.neighborhoods_start_time):
                        end_time = self.neighborhoods_end_time[k]
                        if not start_time <= t < end_time:
                            continue
                        name0, name1, axis = self.neighborhoods[k]
                        tmp_frames = []
                        for tmp_name in (name0, name1):
                            try:
                                tmp_frame = frames[tmp_name]
                            except KeyError:
                                tmp_frame = self.read_vol(tmp_name, i, t_, pool, ex)
                                frames[tmp_name] = tmp_frame
                            tmp_frames.append(tmp_frame)

                        crop0, crop1 = crop_for_alignment(
                            *tmp_frames, axis, self.shift_px
                        )
                        offset = fft_translation_3d(crop0, crop1)
                        offset[axis] += self.shift_px
                        self.offsets_neighbors[t, k] = offset

                if i + 1 < len(self.config.files):
                    for start_name in self.start_names_by_file[i + 1]:
                        if start_name in tmp_start_names:
                            continue
                        frame = self.read_vol(start_name, i, self.nts[i] - 1, pool, ex)
                        prev_fft[start_name] = spfft.rfftn(
                            frame[slices], workers=workers_per_fft
                        )

            current_start_time += self.nts[i]
            close_handle_pool(pool)

        calced_start_times = [
            int(np.where(np.any(self.offsets_neighbors[::1, i] != 0, axis=1))[0][0])
            for i in range(self.offsets_neighbors.shape[1])
        ]
        calced_end_times = [
            self.nt
            - int(np.where(np.any(self.offsets_neighbors[::-1, i] != 0, axis=1))[0][0])
            for i in range(self.offsets_neighbors.shape[1])
        ]
        self.save()
        assert self.neighborhoods_start_time == calced_start_times
        assert self.neighborhoods_end_time == calced_end_times

    def manual_realignment(self, workers_per_fft=-1):
        for start_name, times in self.config.manual_realignment_time.items():
            s_ind = self.start_names.index(start_name)

            for i in tqdm(times):
                slices = get_slices(self.config.realignment_slices, self.min_nz)

                frame1 = self.read_vol_simple(i, start_name)[slices]
                frame0 = self.read_vol_simple(i - 1, start_name)[slices]

                self.offsets_time[i - 1, s_ind] = fft_translation_3d(
                    frame0, frame1, workers=workers_per_fft
                )

        self.save()

    def final_coordinates(self):
        start_name = self.start_names_by_file[0][0]
        self.global_coordinates = {(0, start_name): np.zeros(3)}

        file_i = 0
        for t in range(self.nt):
            if t >= self.max_t_by_file[file_i]:
                file_i += 1
            start_names = self.start_names_by_file[file_i]
            if t > 0:  # für t=0 wird start_name bei 0,0,0 platziert
                for start_name in start_names:
                    s_ind = self.start_names.index(start_name)
                    self.global_coordinates[(t, start_name)] = (
                        self.offsets_time[t - 1, s_ind]
                        + self.global_coordinates[(t - 1, start_name)]
                    )
                    # Da jetzt auch start_name einen
                    # Offset relativ zu t=0 hat, wird
                    # dieser hier hinzugefügt.
            neighbors = [(i, j, k) for k, (i, j, _) in enumerate(self.neighborhoods)]

            while len(neighbors) > 0:
                # Solange die Liste mit noch nicht platzierten Neighbors nicht leer ist,
                # werden die relativen Offsets zwischen
                # Neighbors in die global_coordinates eingefüllt,
                # wenn ein Neighbor vorhanden ist.
                i, j, k = neighbors.pop()
                if (
                    not self.neighborhoods_start_time[k]
                    <= t
                    < self.neighborhoods_end_time[k]
                ):
                    continue
                offset = self.offsets_neighbors[t, k]
                if (t, i) in self.global_coordinates:
                    self.global_coordinates[(t, j)] = (
                        self.global_coordinates[(t, i)] + offset
                    )
                elif (t, j) in self.global_coordinates:
                    self.global_coordinates[(t, i)] = (
                        self.global_coordinates[(t, j)] - offset
                    )
                else:
                    # Falls weder i noch j bereits platziert sind,
                    # werden sie wieder zu neighbors hinzugefügt
                    neighbors.insert(0, (i, j, k))

    def blend(self, filepath=None, t0=-1, t1=None, workers=10):
        if t1 is None:
            t1 = self.nt
        try:
            global_coordinates = self.global_coordinates
        except AttributeError:
            self.final_coordinates()
            global_coordinates = self.global_coordinates

        coords_np = np.array(list(global_coordinates.values()))
        min_extent = coords_np.min(axis=0)
        max_extent = coords_np.max(axis=0) + np.array((self.min_nz, self.ny, self.nx))
        extent = np.stack((min_extent, max_extent), axis=1)
        out_z, out_y, out_x = (extent[:, 1] - extent[:, 0]).astype(int)

        chunk_shape = (1, 32, 512, 512)
        shape = (self.nt, out_z, out_y, out_x)
        _, czs, cys, cxs = np.array(shape) // np.array(chunk_shape) + 1
        _, cz, cy, cx = chunk_shape

        dtype = self.nd2_files[0].dtype
        if filepath is None:
            filepath = self.path_canvas
        canvas = zarr.open(
            filepath,
            mode="a",
            shape=shape,
            chunks=chunk_shape,
            dtype=dtype,
        )
        accum = np.zeros((out_z, out_y, out_x), dtype=np.float32)
        print(accum.shape)
        print(canvas.shape)
        counts = np.zeros((out_z, out_y, out_x), dtype=np.float32)
        inds = np.zeros((out_z, out_y, out_x), dtype=bool)

        current_start_time = 0
        for i, file in enumerate(self.config.files):
            pool = open_handle_pool(file, workers)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for t_ in trange(self.nts[i]):
                    t = t_ + current_start_time
                    if not t0 < t < t1:
                        continue
                    accum[:] = 0
                    counts[:] = 0
                    inds[:] = False
                    for name in self.names_list:
                        if not self.time_offsets[name] <= t < self.end_times[name]:
                            continue
                        frame = self.read_vol(name, i, t_, pool, ex)

                        offset = (global_coordinates[(t, name)] - extent[:, 0]).astype(
                            int
                        )
                        z0, y0, x0 = offset
                        sls = (
                            slice(z0, z0 + self.min_nz),
                            slice(y0, y0 + self.ny),
                            slice(x0, x0 + self.nx),
                        )

                        weights_y = np.ones(frame.shape[1], dtype=np.float32)
                        weights_x = np.ones(frame.shape[2], dtype=np.float32)
                        weights_ = [weights_y, weights_x]

                        for k, (ii, jj, axis) in enumerate(self.neighborhoods):
                            if (
                                not self.neighborhoods_start_time[k]
                                <= t
                                < self.neighborhoods_end_time[k]
                            ):
                                continue
                            if ii == name:
                                diff = (
                                    global_coordinates[(t, jj)]
                                    - global_coordinates[(t, name)]
                                ).astype(int)
                                length = diff[axis]
                                weights_[axis - 1][-length:] = np.linspace(
                                    0.01, 0.99, length
                                )[::-1]
                            if jj == name:
                                diff = (
                                    global_coordinates[(t, name)]
                                    - global_coordinates[(t, ii)]
                                ).astype(int)
                                length = diff[axis]
                                weights_[axis - 1][:length] = np.linspace(
                                    0.01, 0.99, length
                                )

                        weights = weights_x[np.newaxis, :] * weights_y[:, np.newaxis]

                        accum[sls] += frame * weights
                        counts[sls] += weights
                        inds[sls] = True

                    np.divide(accum, counts, out=accum, where=inds)
                    for z in range(czs):
                        for y in range(cys):
                            for x in range(cxs):
                                sl = (
                                    t,
                                    slice(z * cz, (z + 1) * cz),
                                    slice(y * cy, (y + 1) * cy),
                                    slice(x * cx, (x + 1) * cx),
                                )
                                if np.any(inds[sl[1:]]):
                                    try:
                                        canvas[sl] = accum[sl[1:]].astype(dtype)
                                    except ValueError as e:
                                        print(canvas[sl].shape)
                                        print(accum[sl[1:]].shape)
                                        raise e

            current_start_time += self.nts[i]
            close_handle_pool(pool)
