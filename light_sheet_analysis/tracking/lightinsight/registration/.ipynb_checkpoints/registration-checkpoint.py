import numpy as np
import skimage
import zarr
from joblib import Parallel, delayed
from ome_zarr.writer import write_image, write_labels


def calc_translation(
    i,
    leading_channel,
    percentile=None,
    median_blurr=None,
    upsample_factor=3,
    num_comparisons=1,
    sobel=False,
    register_3D=False,
    weighted_average=False,
    metric="max",
    method="phase_cross_correlation",
    mask_channel=None,
    pyramid_order="0",
):

    if mask_channel != None:
        all_masks_moving = mask_channel[i] > 0
        moving = leading_channel[str(i)][pyramid_order][:] * all_masks_moving
    else:
        moving = leading_channel[str(i)][pyramid_order][:]

    num_comparisons = min([i, num_comparisons])

    if register_3D:
        if mask_channel != None:
            all_masks_target = mask_channel[i - 1] > 0
            target = leading_channel[str(i - 1)][pyramid_order][:] * all_masks_target

        else:
            target = leading_channel[str(i - 1)][pyramid_order][:]

        if median_blurr != None:
            target = skimage.filters.median(
                target, footprint=skimage.morphology.disk(median_blurr)
            )
            moving = skimage.filters.median(
                moving, footprint=skimage.morphology.disk(median_blurr)
            )

        if percentile != None:
            moving = np.interp(
                moving,
                (
                    np.percentile(moving, percentile),
                    np.percentile(moving, 100 - percentile),
                ),
                (0, 65535),
            )
            target = np.interp(
                target,
                (
                    np.percentile(target, percentile),
                    np.percentile(target, 100 - percentile),
                ),
                (0, 65535),
            )
        if sobel:
            moving = skimage.filters.sobel(moving)
            target = skimage.filters.sobel(target)

        if method == "phase_cross_correlation":
            translation_all = [
                skimage.registration.phase_cross_correlation(
                    target, moving, upsample_factor=upsample_factor, normalization=None
                )[0]
            ]

        elif method == "optical_flow_tvl1":
            v, u, w = skimage.registration.optical_flow_tvl1(
                target,
                moving,
                prefilter=True,
                num_warp=25,
                tightness=32,
                num_iter=10,
                attachment=64,
            )
            translation_all = [
                np.median(v[target != 0]),
                np.median(u[target != 0]),
                np.median(w[target != 0]),
            ]

        translation_avg[1:] = translation_avg[1:] * (2 ** (int(pyramid_order)))

    else:
        translation_all = []
        for k in range(1, num_comparisons + 1):
            ax_cont = [[1, 2], [0, 2], [0, 1]]
            axis = [0, 1, 2]
            if mask_channel != None:
                all_masks_target = leading_channel[str(i - k)][pyramid_order][:] > 0
                target = (
                    leading_channel[str(i - k)][pyramid_order][:] * all_masks_target
                )
            else:
                target = leading_channel[str(i - k)][pyramid_order][:]

            translation = []
            for j in axis:

                if metric == "max":
                    target_axis = target.max(j)
                    moving_axis = moving.max(j)
                if metric == "mean":
                    target_axis = target.mean(j)
                    moving_axis = moving.mean(j)

                if median_blurr != None:
                    target_axis = skimage.filters.median(
                        target_axis, footprint=skimage.morphology.disk(median_blurr)
                    )
                    moving_axis = skimage.filters.median(
                        moving_axis, footprint=skimage.morphology.disk(median_blurr)
                    )

                if percentile != None:
                    moving_axis = np.interp(
                        moving_axis,
                        (
                            np.percentile(moving_axis, percentile),
                            np.percentile(moving_axis, 100 - percentile),
                        ),
                        (0, 65535),
                    )
                    target_axis = np.interp(
                        target_axis,
                        (
                            np.percentile(target_axis, percentile),
                            np.percentile(target_axis, 100 - percentile),
                        ),
                        (0, 65535),
                    )
                if sobel:
                    moving_axis = skimage.filters.sobel(moving_axis)
                    target_axis = skimage.filters.sobel(target_axis)

                if method == "phase_cross_correlation":
                    translation.append(
                        skimage.registration.phase_cross_correlation(
                            target_axis,
                            moving_axis,
                            upsample_factor=upsample_factor,
                            normalization=None,
                        )[0]
                    )
                elif method == "optical_flow_tvl1":

                    v, u = skimage.registration.optical_flow_tvl1(
                        moving_axis,
                        target_axis,
                        prefilter=True,
                        num_warp=25,
                        tightness=32,
                        num_iter=10,
                        attachment=64,
                    )
                    # v,u= skimage.registration.optical_flow_ilk(moving_axis,target_axis,prefilter=True,radius=80)
                    translation.append(
                        [np.median(v[target_axis != 0]), np.median(u[target_axis != 0])]
                    )

            translation_avg = np.zeros((3,))
            for ax, result_one in zip(ax_cont, translation):
                for axis, res in zip(ax, result_one):
                    translation_avg[axis] += res

            translation_avg[1:] = translation_avg[1:] * (2 ** (int(pyramid_order)))

            if weighted_average:
                translation_all.append((translation_avg / 2) * (1 / k))
            else:
                translation_all.append(translation_avg / 2)

    return translation_all


def calculate_cummulative_translations(
    leading_channel: zarr.array = None,
    percentile: int = 30,
    num_comparisons: int = 1,
    upsample_factor: int = 3,
    weighted_average: bool = False,
    register_3D: bool = False,
    median_blurr: int = None,
    metric: str = "max",
    sobel: bool = False,
    method="phase_cross_correlation",
    mask_channel=None,
    pyramid_order="0",
    verbose: int = 2,
    n_jobs: int = 2,
    time_points: np.array = None,
):
    phase_translations = Parallel(
        n_jobs=n_jobs, backend="multiprocessing", verbose=verbose
    )(
        delayed(calc_translation)(
            i,
            leading_channel=leading_channel,
            percentile=percentile,
            num_comparisons=num_comparisons,
            upsample_factor=upsample_factor,
            weighted_average=weighted_average,
            register_3D=register_3D,
            median_blurr=median_blurr,
            sobel=sobel,
            metric=metric,
            method=method,
            mask_channel=mask_channel,
            pyramid_order=pyramid_order,
        )
        for i in time_points[1:]
    )

    if weighted_average:
        all_translations = np.zeros(
            (len(phase_translations), len(phase_translations), 3)
        )
        all_translations[:] = np.nan
        for i in range(1, len(phase_translations) + 1):
            num_comparison = min([i, num_comparisons])
            all_translations[i - 1, (i - (num_comparison)) : i, :] = np.flip(
                np.diff(
                    np.array(phase_translations[i - 1]),
                    axis=0,
                    prepend=np.array([[0, 0, 0]]),
                ),
                0,
            )
        translations = np.nansum(all_translations, 0)
        weight = 0
        for i in range(1, num_comparisons + 1):
            weight += 1 / i
        translations = translations * (1 / weight)
    else:
        all_translations = np.zeros(
            (len(phase_translations), len(phase_translations), 3)
        )
        all_translations[:] = np.nan
        for i in range(1, len(phase_translations) + 1):
            num_comparison = min([i, num_comparisons])
            all_translations[i - 1, (i - (num_comparison)) : i, :] = np.flip(
                np.diff(
                    np.array(phase_translations[i - 1]),
                    axis=0,
                    prepend=np.array([[0, 0, 0]]),
                ),
                0,
            )
        translations = np.nanmean(all_translations, 0)

    cum_sum_translations = np.round(np.cumsum(translations, 0)).astype(int)
    min_all = cum_sum_translations.min(0)
    max_all = cum_sum_translations.max(0)
    padding = []
    for i in range(len(min_all)):
        padding.append([abs(min_all[i]), abs(max_all[i])])
    return cum_sum_translations, padding


def register_frame(
    time_point, input_movie, channel_group, padding, chunk_size, cum_sum_translations
):
    axis = [0, 1, 2]
    if time_point == 0:

        registered = np.pad(input_movie[str(time_point)]["0"][:], padding)
    else:
        registered = np.pad(input_movie[str(time_point)]["0"][:], padding)
        # registered=np.roll(registered, cum_sum_translations[time_point-1])
        translation = cum_sum_translations[time_point - 1]
        for ax in axis:
            registered = np.roll(registered, translation[ax], axis=ax)
    time_group = channel_group.require_group(str(time_point))
    write_image(
        image=registered,
        group=time_group,
        axes="zyx",
        storage_options=dict(chunks=chunk_size),
    )


def register_labels(
    time_point,
    label,
    input_movie,
    channel_group,
    padding,
    chunk_size,
    cum_sum_translations,
):
    axis = [0, 1, 2]
    if time_point == 0:

        registered = np.pad(
            input_movie[str(time_point)]["labels"][label]["0"][:], padding
        )
    else:
        registered = np.pad(
            input_movie[str(time_point)]["labels"][label]["0"][:], padding
        )
        # registered=np.roll(registered, cum_sum_translations[time_point-1])
        translation = cum_sum_translations[time_point - 1]
        for ax in axis:
            registered = np.roll(registered, translation[ax], axis=ax)
    time_group = channel_group.require_group(str(time_point))
    write_labels(
        labels=registered,
        group=time_group,
        name=label,
        axes="zyx",
        storage_options=dict(chunks=chunk_size),
    )
