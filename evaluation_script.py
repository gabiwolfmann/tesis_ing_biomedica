import pandas as pd
from sklearn.cluster import DBSCAN
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evolution in time")
    parser.add_argument(
        "old_csv",
        type=str,
        help=(
            "Full path to the directory having the csv file of older image. E.g. "
            "`/home/app/src/data/evolution/patient_1_old.csv`."
        ),
    )

    parser.add_argument(
        "new_csv",
        type=str,
        help=(
            "Full path to the directory having csv file of newer image. E.g. "
            "`/home/app/src/data/evolution/patient_1_new.csv`."
        ),
    )
    args = parser.parse_args()

    return args


def main(path_old, path_new):
    df_old = pd.read_csv(path_old)
    df_new = pd.read_csv(path_new)

    df_union = pd.concat([df_old, df_new], axis=0).reset_index(drop=True)
    df_union.rename(columns={"Unnamed: 0": "Cluster_number"}, inplace=True)

    db_df = df_union[
        ["relative_coordinates_centroid_x", "relative_coordinates_centroid_y"]
    ]
    clustering = DBSCAN(eps=0.05, min_samples=2).fit(db_df)

    labels = clustering.labels_
    df_union["label_cluster"] = clustering.labels_

    for label in range(
        df_union.shape[0] // 2
    ):  # maximum number of clusters that an image can have
        try:
            df = df_union[df_union["label_cluster"] == label]
            if not df.empty:
                difference_number_microc = sum(
                    df.number_microcalcification.loc[df_old.shape[0] :]
                ) - sum(df.number_microcalcification.loc[: df_old.shape[0] - 1])
                difference_area = sum(df.area_mm2.loc[df_old.shape[0] :]) - sum(
                    df.area_mm2.loc[: df_old.shape[0] - 1]
                )

                print(
                    f"El cluster color {df.color.loc[0:df_old.shape[0]-1].values} de la imagen previa tiene {sum(df.number_microcalcification.loc[0:df_old.shape[0]-1])} microcalcificaciones abarcando un área de {sum(df.area_mm2.loc[0:df_old.shape[0]-1])} mm2, y el cluster color {df.color.loc[df_old.shape[0]:].values} de la imagen nueva tiene {sum(df.number_microcalcification.loc[df_old.shape[0]:])} microcalcificaciones abarcando un área de {sum(df.area_mm2.loc[df_old.shape[0]:])} mm2"
                )
                print(
                    f"El número de microcalcificaciones creció: {difference_number_microc}"
                )
                print(f"El tamaño del cluster creció {round(difference_area,2)} mm2\n")

        except:
            continue
    print(
        f"número total de microcalcificaciones de la imagen antigua: {sum(df_old.number_microcalcification)}"
    )
    print(
        f"número total de microcalcificaciones de la imagen nueva: {sum(df_new.number_microcalcification)}\n"
    )
    print(
        f"En total el cambio fue de {sum(df_new.number_microcalcification)-sum(df_old.number_microcalcification)} microcalcificaciones"
    )


if __name__ == "__main__":
    args = parse_args()
    main(args.old_csv, args.new_csv)
