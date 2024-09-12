import os

from absl import app, flags
from color_movie_generator import create_movie

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "input_folder",
    "/cluster/work/treutlein/DATA/imaging/viventis/Position_5_Settings_1_denoised_deconvolved_better_preprocessing_cropped_registered/",
    "Input folder with the TIFF stacks",
)
flags.DEFINE_string(
    "output_folder",
    "/cluster/home/gutgi/",
    "Creates output folder (if it doesn't already exist) for the different movies",
)
flags.DEFINE_string("f", "", "kernel")

flags.DEFINE_string("wait", "", "wait_argument")

flags.DEFINE_string("jobname", "", "jobname")


def main(argv):
    input_dir = FLAGS.input_folder
    output_dir = FLAGS.output_folder

    # Number of all files
    list = os.listdir(input_dir + "/GFP/")
    number_files = len(list)
    name_folder = input_dir.split("/")[-2]

    # Create MIP over movie
    output_name = output_dir + "/" + name_folder + "_stack_all_projection.mov"
    create_movie(
        input_dir,
        output_name,
        time_point_start=1,
        time_point_stop=number_files,
        start_angle=1,
        stop_angle=1,
        n_frames=number_files,
        scale=1,
        voxel_sizes=[0.347, 0.347, 0.347],
        attenuation=False,
        MIP=True,
        stack_slice=None,
        run_through_slice=False,
        pad_size=0,
    )

    # Rotation over whole movie
    output_name = output_dir + "/" + name_folder + "_stack_rotation.mov"
    create_movie(
        input_dir,
        output_name,
        time_point_start=1,
        time_point_stop=number_files,
        start_angle=1,
        stop_angle=360,
        n_frames=number_files,
        scale=0.25,
        voxel_sizes=[2, 0.347, 0.347],
        attenuation=True,
        MIP=True,
        stack_slice=None,
        run_through_slice=False,
    )

    # Slice and rotation for t=1,0.5*max,max
    times = [1, int(0.5 * number_files), number_files]
    print(times)
    # Slice overtime
    for time in times:
        # One slice
        output_name = (
            output_dir + "/" + name_folder + "_one_slice_time_" + str(time) + ".mov"
        )
        create_movie(
            input_dir,
            output_name,
            time_point_start=time,
            time_point_stop=time,
            start_angle=1,
            stop_angle=1,
            n_frames=5,
            scale=0.25,
            voxel_sizes=[2, 0.347, 0.347],
            attenuation=False,
            MIP=False,
            stack_slice=None,
            run_through_slice=True,
        )

        # Create Rotation one stack
        output_name = (
            output_dir + "/" + name_folder + "_one_rotation_time_" + str(time) + ".mov"
        )
        create_movie(
            input_dir,
            output_name,
            time_point_start=time,
            time_point_stop=time,
            start_angle=1,
            stop_angle=360,
            n_frames=360,
            scale=0.25,
            voxel_sizes=[2, 0.347, 0.347],
            attenuation=True,
            MIP=True,
            stack_slice=None,
            run_through_slice=False,
        )


if __name__ == "__main__":
    app.run(main)
