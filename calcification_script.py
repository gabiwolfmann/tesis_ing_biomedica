import utils
import argparse
import os

import warnings
warnings.filterwarnings("ignore")



def parse_args():
    parser = argparse.ArgumentParser(description="Find calcification.")
    parser.add_argument(
        "image_path",
        type=str,
        help=(
            "Full path to the directory having the dicom image. E.g. "
            "`/home/app/src/data/patient_1.dcm`."
        ),
    )

    parser.add_argument(
        "percentage",
        type=float,
        nargs='?',
        default=0.7,
        help=(
            "Percentage to binarize the image. E.g. "
            "0.7"
        ),
    )

    parser.add_argument(
        "region_growing",
        type=str,
        nargs='?',
        default='True',
        help=(
            "If it uses region growing or polygon adjust to extract pectoral muscle. E.g. "
            "True"
        ),
    )


    args = parser.parse_args()

    return args


def main(path, percentage ,region_growing):

    #create folder with patien name
    root_dir = "./data/evolution"
    basename = os.path.basename(path)
    filename, ext = os.path.splitext(basename)
    patient_name=filename.split('_')[0]
    folder_path=os.path.join(root_dir, patient_name)
    os.makedirs(folder_path, exist_ok=True)

    print(f'filename: {filename}')
    imagen_2D, image_array, max_array, data = utils.open(path)

    crop_image, array_crop_image = utils.crop(imagen_2D, image_array,patient_name,filename)
    print("crop succesfully \n")

    img_sin_piel, array_img_sin_piel = utils.extract_skin(crop_image, array_crop_image)
    print("extract_skin succesfully \n")

    if region_growing == 'True':


        img_sin_musculo, array_img_sin_musculo = utils.extract_muscle_region_growing(
            img_sin_piel, array_img_sin_piel, data
        )
        

    else:
        img_sin_musculo, array_img_sin_musculo = utils.extract_muscle_polygon(
            img_sin_piel, array_img_sin_piel, data
        )
    filtered_props_df, label_image, list_labels_filtered = utils.find_calcification(
        array_img_sin_musculo, max_array, percentage
    )

    print("find_calcification succesfully\n")

    list_colors = [
        "lime",
        "red",
        "blue",
        "yellow",
        "orange",
        "fuchsia",
        "saddlebrown",
        "cyan",
        "violet",
        "white",
        "green",
        "pink",
    ]

    db_df, clustering = utils.clustering(filtered_props_df)

    utils.show_clusters(
            label_image,
            list_labels_filtered,
            filtered_props_df,
            list_colors,
            db_df,
            patient_name,
            filename,
        )

    results = utils.create_df(
        array_img_sin_piel, array_crop_image, data, list_colors, db_df, clustering
    )

    #create csv with the results
    csv_name = filename + ".csv"
    output_file = os.path.join(folder_path, csv_name)
    results.to_csv(output_file)


if __name__ == "__main__":
    args = parse_args()
    main(args.image_path,args.percentage,args.region_growing)
