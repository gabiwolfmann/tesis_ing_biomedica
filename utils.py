import matplotlib.pyplot as plt
from PIL import Image
import SimpleITK as sitk
import numpy as np
import pydicom
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean
from scipy.spatial import ConvexHull

# from skimage import data
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk
from skimage.measure import regionprops_table
from skimage.draw import polygon
import matplotlib.patches as mpatches
import os


def open(path):
    imagen = sitk.ReadImage(path)
    data = pydicom.dcmread(path)
    array_imagen = sitk.GetArrayViewFromImage(imagen)  
    image_array = array_imagen[0, :, :]  # Remove first dimension

    imagen_2D = sitk.GetImageFromArray(image_array)
    max_array = image_array.max()
    print("Open succesfully \n")

    return imagen_2D, image_array, max_array, data


def crop(imagen_2D, image_array,patient_name,filename):
    print("start cropping \n")
    bina = imagen_2D > 0
    label_image = label(sitk.GetArrayFromImage(bina))
    props_table = regionprops_table(
        label_image,
        sitk.GetArrayFromImage(imagen_2D),
        properties=["area", "label", "bbox"],
    )
    props_df = pd.DataFrame(props_table)
    index_max = props_df.area.argmax()
    filamin = props_df["bbox-0"].loc[index_max]
    columnamin = props_df["bbox-1"].loc[index_max]
    filamax = props_df["bbox-2"].loc[index_max]
    columnamax = props_df["bbox-3"].loc[index_max]
    array_crop_image = image_array[filamin:filamax, columnamin:columnamax]
    crop_image = sitk.GetImageFromArray(array_crop_image)
    

    root='./data/evolution'

    img_name=filename +'crop_image.png'
    output_path=os.path.join(root,patient_name,img_name)
    Image.fromarray(np.uint16(array_crop_image*16)).save(output_path)

    return crop_image, array_crop_image


def extract_skin(crop_image, array_crop_image):
    bina = crop_image > 0  # binarize full image
    bina_v = sitk.GetArrayFromImage(bina)
    contorno = sitk.BinaryContour(
        bina, fullyConnected=True, backgroundValue=0.0, foregroundValue=1.0
    )

    contorno = sitk.BinaryDilate(
        contorno, [35, 35]
    )  # dilates the contour to give it greater thickness and remove all the skin

    contornoNegativo = (
        contorno == 0
    )  # is the negative of the contour. It is used for multiplication with the binary image
    arrayContornoNegativo = sitk.GetArrayFromImage(contornoNegativo)
    arrayBinariaSinPiel = bina_v * arrayContornoNegativo
    array_img_sin_piel = (
        array_crop_image * arrayBinariaSinPiel
    )  # multiplies the original array by the unskinned binary, 
    #leaving the original unskinned (since it multiplies by 1 and 0)
    img_sin_piel = sitk.GetImageFromArray(array_img_sin_piel)

    return img_sin_piel, array_img_sin_piel


def extract_muscle_region_growing(img_sin_piel, array_img_sin_piel, data):
    """"
            Segment pixels with similar statistics using connectivity.

            This filter extracts a connected set of pixels whose pixel intensities
            are consistent with the pixel statistics of a seed point. The mean and
            variance across a neighborhood (8-connected, 26-connected, etc.) are
            calculated for a seed point. Then pixels connected to this seed point
            whose values are within the confidence interval for the seed point are
            grouped. The width of the confidence interval is controlled by the
            "Multiplier" variable (the confidence interval is the mean plus or
            minus the "Multiplier" times the standard deviation). If the
            intensity variations across a segment were gaussian, a "Multiplier"
            setting of 2.5 would define a confidence interval wide enough to
            capture 99% of samples in the segment.

            After this initial segmentation is calculated, the mean and variance
            are re-calculated. All the pixels in the previous segmentation are
            used to calculate the mean the standard deviation (as opposed to using
            the pixels in the neighborhood of the seed point). The segmentation is
            then recalculated using these refined estimates for the mean and
            variance of the pixel values. This process is repeated for the
            specified number of iterations. Setting the "NumberOfIterations" to
            zero stops the algorithm after the initial segmentation from the seed
            point.

    """
    #  I only extract if it has muscle
    if data.ProtocolName[2:5] == "MLO":
        size = img_sin_piel.GetSize()
        max_col = size[0]
        max_row = size[1]


        # Depending on which side is the muscle generated the seed
        if data.ProtocolName == "L MLO":
            seedList_2 = [(0, 50), (50, 100), (max_col // 8, 100), (50, max_row // 4)]
        else:
            seedList_2 = [
                (max_col - 1, 20),
                (max_col - 1, 100),
                (max_col - 100, max_row // 4),
            ]

        fin = sitk.ConfidenceConnected(
            img_sin_piel,
            seedList_2,
            numberOfIterations=1,
            multiplier=2.5,
            initialNeighborhoodRadius=1,
            replaceValue=1,
        )


        fin = sitk.BinaryDilate(fin, [15, 15])
        fin = sitk.BinaryErode(fin, [15, 15]) 
        neg = fin == 0
        neg_a = sitk.GetArrayFromImage(neg)
        array_img_sin_musculo = neg_a * array_img_sin_piel
        img_sin_musculo = sitk.GetImageFromArray(array_img_sin_musculo)

        print("extract_muscle_region_growing succesfully \n")

    else:
        array_img_sin_musculo = array_img_sin_piel
        img_sin_musculo = img_sin_piel

    return img_sin_musculo, array_img_sin_musculo


def extract_muscle_polygon(img_sin_piel, array_img_sin_piel, data):
    if data.ProtocolName[2:5] == "MLO":
        contorno_sin_piel = sitk.BinaryContour(
            img_sin_piel, fullyConnected=False, backgroundValue=0.0, foregroundValue=1.0
        )
        """
            takes a binary image as input, where the pixels in the objects are
            the pixels with a value equal to ForegroundValue. Only the pixels on
            the contours of the objects are kept. The pixels not on the border are
            changed to BackgroundValue.

            Realiza el borde morfologico de la imagen dilatada - original para quedarse con el borde externo.
        """
        contorno_sin_piel_v = sitk.GetArrayViewFromImage(contorno_sin_piel)
        coordenadas_sin_piel = np.where(
            contorno_sin_piel_v
        )

        if data.ProtocolName[0] == "L":
            size_sin_piel = img_sin_piel.GetSize()
            columnamin_sin_piel = 0
            filamin_sin_piel = coordenadas_sin_piel[0].min()
            columnamax_sin_piel = coordenadas_sin_piel[1].max()
            filamax_sin_piel = coordenadas_sin_piel[0].max()

            r = np.array([1, filamax_sin_piel // 2.5, filamax_sin_piel // 1.5, 1])
            c = np.array(
                [
                    columnamax_sin_piel // 1.5,
                    columnamax_sin_piel // 3.5,
                    columnamin_sin_piel,
                    columnamin_sin_piel,
                ]
            )
            rr, cc = polygon(r, c)
            array_img_sin_musculo = array_img_sin_piel.copy()
            array_img_sin_musculo[rr, cc] = 0
            img_sin_musculo = sitk.GetImageFromArray(array_img_sin_musculo)

        else:
            # Same reasoning, but since the image is on the other side, we are not 
            # really looking for the maximum column, but the minimum
            size_sin_piel = img_sin_piel.GetSize()
            columnamin_sin_piel = coordenadas_sin_piel[1].min()
            filamin_sin_piel = coordenadas_sin_piel[0].min()
            columnamax_sin_piel = size_sin_piel[0]
            filamax_sin_piel = coordenadas_sin_piel[0].max()

            r = np.array([1, filamax_sin_piel // 2.5, filamax_sin_piel // 1.5, 1])
            c = np.array(
                [
                    columnamax_sin_piel // 2,
                    columnamax_sin_piel // 1.2,
                    columnamax_sin_piel - 1,
                    columnamax_sin_piel - 1,
                ]
            )
            rr, cc = polygon(r, c)
            array_img_sin_musculo = array_img_sin_piel.copy()
            array_img_sin_musculo[rr, cc] = 0
            img_sin_musculo = sitk.GetImageFromArray(array_img_sin_musculo)

        print("extract_muscle_polygon succesfully \n")
    else:
        array_img_sin_musculo = array_img_sin_piel
        img_sin_musculo = img_sin_piel

    return img_sin_musculo, array_img_sin_musculo


def find_calcification(array_img_sin_musculo, max_array, percentage):
    umbral = max_array * percentage
    bw = closing(array_img_sin_musculo > umbral, disk(3))
    # remove artifacts connected to image border
    cleared = clear_border(bw)
    # label image regions
    label_image = label(cleared)
    props_table = regionprops_table(
        label_image,
        array_img_sin_musculo,
        properties=[
            "area",
            "equivalent_diameter",
            "label",
            "bbox",
            "perimeter",
            "eccentricity",
            "centroid",
            "min_intensity",
            "max_intensity",
            "mean_intensity",
        ],
    )
    props_df = pd.DataFrame(props_table)
    # filter in a lower and upper way to get only the microcalcifications
    area_threshold_high = 40
    area_threshold_low = 1
    eccentricity_threshold = 0.9
    filtered_props_df = props_df[(props_df["area"] < area_threshold_high)]
    filtered_props_df = filtered_props_df[
        (filtered_props_df["eccentricity"] < eccentricity_threshold)
    ]
    filtered_props_df = filtered_props_df[
        (filtered_props_df["area"] > area_threshold_low)
        | (filtered_props_df["max_intensity"] > (umbral * 1.05))
    ]
    list_labels_filtered = filtered_props_df["label"].tolist()
    number_calcification = filtered_props_df.shape[0]
    print(f"the number of microcalcifications found before clustering is: {number_calcification}\n")

    return filtered_props_df, label_image, list_labels_filtered


def eccentricity(hull):
    points = hull.points

    small_latwise = np.min(points[points[:, 0] == np.min(points[:, 0])], 0)
    small_lonwise = np.min(points[points[:, 1] == np.min(points[:, 1])], 0)
    big_latwise = np.max(points[points[:, 0] == np.max(points[:, 0])], 0)
    big_lonwise = np.max(points[points[:, 1] == np.max(points[:, 1])], 0)
    distance_lat = euclidean(big_latwise, small_latwise)
    distance_lon = euclidean(big_lonwise, small_lonwise)
    if distance_lat >= distance_lon:
        major_axis_length = distance_lat
        minor_axis_length = distance_lon
    else:
        major_axis_length = distance_lon
        minor_axis_length = distance_lat
    a = major_axis_length / 2
    b = minor_axis_length / 2
    ecc = np.sqrt(np.square(a) - np.square(b)) / a
    return ecc.round(2)


def clustering(filtered_props_df):
    """
    eps= The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples= The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.

    If pixel spacing its 0.07 mm, 100 pixels it will be 7 mm. 
    """
    # use the centroid to generate the different clusters in the microcalcification
    db_df = filtered_props_df[["centroid-0", "centroid-1"]]
    clustering = DBSCAN(eps=90, min_samples=4).fit(db_df)

    db_df["label_cluster"] = clustering.labels_

    return db_df, clustering


def show_clusters(
    label_image,
    list_labels_filtered,
    filtered_props_df,
    list_colors,
    db_df,
    patient_name,
    filename,
):

    root='./data/evolution'
    img_name=filename+'crop_image.png'
    input_path=os.path.join(root,patient_name,img_name)

    cluster_image=filename +'cluster_image.png'
    output_path=os.path.join(root,patient_name,cluster_image)

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, figsize=(7, 15))

    image=plt.imread(input_path)
    plt.imshow(image,cmap='gray')


    number_total_calcifications_in_clusters = 0
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.label in list_labels_filtered:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox

            index_value = filtered_props_df.index[
                filtered_props_df["label"] == region.label
            ].tolist()
            # take the index of the label
            label_cluster = db_df["label_cluster"].loc[index_value[0]]

            # only view the microcalcifications that there are inside a cluster
            if label_cluster != -1:

                # use this index to get the cluster label. Then use the different colors for each cluster.

                rect = mpatches.Rectangle(
                    (minc, minr),
                    maxc - minc,
                    maxr - minr,
                    linewidth=5,
                    edgecolor=list_colors[label_cluster],
                    facecolor="none",
                )

                ax.add_patch(rect)

                number_total_calcifications_in_clusters += 1
        
    print(
        f"The number of microcalcifications within clusters found is: {number_total_calcifications_in_clusters} \n"
    )
    plt.tight_layout()
    ax.figure.savefig(output_path)


def create_df(
    array_img_sin_piel, array_crop_image, data, list_colors, db_df, clustering
):
    dictionary_cluster = {}
    mm2 = data.PixelSpacing[0] * data.PixelSpacing[0]

    labels = list(set(clustering.labels_))
    for label in labels[0:-1]:
        cluster_points = db_df[["centroid-0", "centroid-1"]][
            (db_df["label_cluster"] == label)
        ]
        hull = ConvexHull(cluster_points)
        cx = np.mean(hull.points[hull.vertices, 0]).round()
        cy = np.mean(hull.points[hull.vertices, 1]).round()
        std_cluster = np.std(hull.points, axis=0).round(2)
        ecc = eccentricity(hull)
        number_microcalcification_cluster = hull.npoints
        color = list_colors[label]
        if cx < array_img_sin_piel.shape[0] / 2:
            if cy < array_img_sin_piel.shape[1] / 2:
                ubicacion = "Upper left quadrants"
            else:
                ubicacion = "Upper right quadrants"
        elif cx > array_img_sin_piel.shape[0] / 2:
            if cy < array_img_sin_piel.shape[1] / 2:
                ubicacion = "Lower left quadrants"
            else:
                ubicacion = "Lower right quadrants"

        roundness = 4 * np.pi * hull.volume / (hull.area ** 2)
        if roundness >= 0.95 and roundness < 1.05:
            shape = "Circle"
        elif roundness >= 0.85 and roundness < 0.95:
            shape = "Hexagon"
        elif roundness >= 0.75 and roundness < 0.85:
            shape = "square"
        elif roundness >= 0.65 and roundness < 0.75:
            shape = "rectangle"
        elif roundness >= 0.55 and roundness < 0.65:
            shape = "equilateral triangle"
        elif roundness >= 0.45 and roundness < 0.55:
            shape = "rectangular triangle"
        else:
            shape = "linear"

        dictionary_cluster["cluster %s" % label] = dict(
            {
                "color": color,
                "area_mm2": round(hull.volume * mm2, 2),
                "area_pixel": round(hull.volume, 2),
                "number_microcalcification": number_microcalcification_cluster,
                "eccentricity_cluster": ecc,
                "coordinates_centroid_x": cx,
                "coordinates_centroid_y": cy,
                "relative_coordinates_centroid_x": cx / array_crop_image.shape[0],
                "relative_coordinates_centroid_y": cy / array_crop_image.shape[1],
                "location": ubicacion,
                "std_centroid_x_[mm]": (std_cluster[0] * data.PixelSpacing[0]).round(2),
                "std_centroid_y_[mm]": (std_cluster[1] * data.PixelSpacing[0]).round(2),
                "roundness": round(roundness, 3),
                "shape": shape,
            }
        )

    results = pd.DataFrame.from_dict(dictionary_cluster, orient="index")

    return results
