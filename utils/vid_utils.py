import os
import subprocess




def images_to_mp4(image_dir, output_file, fps=25):
    """
    Converts a sequence of images in a directory to an MP4 video using ffmpeg.

    Args:
        image_dir (str): Path to the directory containing the image sequence.
        output_file (str): Path to the output MP4 video file.
        fps (int): Frames per second for the video.

    Returns:
        str: Success or error message.
    """
    try:
        # Validate the image directory
        if not os.path.exists(image_dir):
            return f"Error: Directory '{image_dir}' does not exist."
        if not os.listdir(image_dir):
            return f"Error: Directory '{image_dir}' is empty."

        # Construct the ffmpeg command
        # Assuming images are named as seq-1.jpg, seq-2.jpg, ...
        input_pattern = os.path.join(image_dir, "seq-%d.jpg")
        command = [
            "ffmpeg",
            "-y",  # Overwrite output if it exists
            "-framerate", str(fps),  # Set frame rate
            "-i", input_pattern,  # Input file pattern
            "-c:v", "libx264",  # Use H.264 codec
            output_file,
        ]

        # Run the ffmpeg command
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check for errors
        if result.returncode != 0:
            return f"Error: ffmpeg failed with the following error:\n{result.stderr}"

        return f"Success: Video saved to '{output_file}'."

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

