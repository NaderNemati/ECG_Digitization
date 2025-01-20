import argparse
import os
import shutil
import hashlib

def get_parser():
    parser = argparse.ArgumentParser(description='Remove hidden data and copy images.')
    parser.add_argument('-i', '--input_folder', type=str, required=True, help='Input folder')
    parser.add_argument('-o', '--output_folder', type=str, required=True, help='Output folder')
    parser.add_argument('--include_images', action='store_true', help='Include images in the output')
    return parser

def calculate_md5(file_path):
    """Calculate the MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def run(args):
    input_folder = args.input_folder
    output_folder = args.output_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            input_file = os.path.join(root, file)
            output_file = os.path.join(output_folder, os.path.relpath(input_file, start=input_folder))

            if args.include_images and file.endswith('.png'):
                if input_file != output_file:
                    if not os.path.exists(os.path.dirname(output_file)):
                        os.makedirs(os.path.dirname(output_file))
                    shutil.copy2(input_file, output_file)
                    print(f"Copied {file} from {input_file} to {output_file} with MD5 checksum: {calculate_md5(output_file)}")
                else:
                    print(f"Skipping file {input_file} because source and destination are the same.")
            elif not args.include_images and not file.endswith('.png'):
                if input_file != output_file:
                    if not os.path.exists(os.path.dirname(output_file)):
                        os.makedirs(os.path.dirname(output_file))
                    shutil.copy2(input_file, output_file)
                    print(f"Copied {file} from {input_file} to {output_file} with MD5 checksum: {calculate_md5(output_file)}")
                else:
                    print(f"Skipping file {input_file} because source and destination are the same.")

if __name__ == "__main__":
    run(get_parser().parse_args())
