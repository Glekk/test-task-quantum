import os
import argparse
from dotenv import load_dotenv
from algorithm import LoFTR, SIFT, DISK_LightGlue
      

def main(args):
    '''
    Main function to run the inference

    Args:
        args: arguments from the command line (alg, path0, path1, img_shape, equalize, clahe, visualize, visualize_non_scaled_images, save)
    '''
    # Load the algorithm
    if args.alg == 'loftr':
        algorithm = LoFTR()
    elif args.alg == 'disk_lg':
        algorithm = DISK_LightGlue()
    elif args.alg == 'sift':
        algorithm = SIFT()
    else:
        raise ValueError(f'Unknown algorithm: {args.algorithm}')

    algorithm(args.path0, args.path1, tuple(map(int, args.img_shape.split(','))), args.no_equalize, args.clahe, args.no_visualize, args.visualize_non_scaled_images, args.save)

    
if __name__ == '__main__':
    # Get the command line arguments
    parser = argparse.ArgumentParser(description='Inference (using large images may take some time)')
    parser.add_argument('--alg', type=str, default='loftr', help='Algorithm to use (loftr, disk_lg, sift)', choices=['loftr', 'disk_lg', 'sift'], required=True)
    parser.add_argument('--path0', type=str, help='Path to the first image', required=True)
    parser.add_argument('--path1', type=str, help='Path to the second image', required=True)
    parser.add_argument('--img_shape', type=str, default='1024,1024', help='Image shape as "height,width"')
    parser.add_argument('--no_equalize', action='store_false', help='Do not equalize the images before processing')
    parser.add_argument('--clahe', action='store_true', help='Apply CLAHE to the images before processing')
    parser.add_argument('--no_visualize', action='store_false', help='Turn off visualization the matches')
    parser.add_argument('--visualize_non_scaled_images', action='store_true', help='Visualize the matches with the original images')
    parser.add_argument('--save', action='store_true', help='Save the visualization')
    args = parser.parse_args()

    main(args)