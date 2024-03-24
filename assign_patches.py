import pandas as pd

from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import columns as c
import geopandas as gpd
from shapely.geometry import box, Polygon
from shapely.affinity import translate
import numpy as np

from copy import deepcopy


class AssignPatchesOperation:
    def _expand_bbox_to_square(self, geometry: Polygon, square_size: int) -> dict:
        """Expands a bounding box to a square of a specified size.

        Args:
            geometry (shapely.geometry): A geometry object.
            square_size (int): The size of the square.

        Returns:
            dict: A dictionary containing the coordinates of the square.
        """
        x_min, y_min, x_max, y_max = geometry.bounds
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        min_x = center_x - square_size / 2
        max_x = center_x + square_size / 2
        min_y = center_y - square_size / 2
        max_y = center_y + square_size / 2
        return {c.MIN_X: min_x, c.MIN_Y: min_y, c.MAX_X: max_x, c.MAX_Y: max_y}

    def _calculate_overflow(self, proposed_coordinates: gpd.GeoDataFrame, image_size) -> gpd.GeoDataFrame:
        """Calculates the overflow of the proposed coordinates.

        NOTE: When we expand the bounding box to a square, the square may overflow the image (extend beyond the image).
        To account for this, we calculate the overflow - the amount by which we should adjust the coordinates of the
        new square to fit within the image.

        Args:
            proposed_coordinates (gpd.GeoDataFrame): A GeoDataFrame containing the proposed coordinates.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing the overflow.
        """
        df_proposed_coordinates = pd.json_normalize(proposed_coordinates)
        negative_overflow = df_proposed_coordinates < 0
        positive_overflow = df_proposed_coordinates > image_size

        negative_adjustment = (df_proposed_coordinates[positive_overflow] - image_size) * -1
        positive_adjustment = df_proposed_coordinates[negative_overflow] * -1

        adjustment = negative_adjustment.combine_first(positive_adjustment)
        x_off = adjustment[c.MAX_X].combine_first(adjustment[c.MIN_X]).fillna(0)
        y_off = adjustment[c.MAX_Y].combine_first(adjustment[c.MIN_Y]).fillna(0)
        df_proposed_coordinates["x_off"] = x_off
        df_proposed_coordinates["y_off"] = y_off
        return df_proposed_coordinates

    def _translate_bboxes_to_fit_within_image(self, df_proposed_coordinates: gpd.GeoDataFrame) -> list:
        """Translates bounding boxes to fit within the image.

        Args:
            df_proposed_coordinates (gpd.GeoDataFrame): A GeoDataFrame containing the proposed coordinates.

        Returns:
            list: A list of bboxes of a new size that fit within the image.
        """
        new_geometries = []
        for row in df_proposed_coordinates.to_dict(orient="records"):
            geometry = box(row[c.MIN_X], row[c.MIN_Y], row[c.MAX_X], row[c.MAX_Y])
            adjusted_geometry = translate(geometry, xoff=row["x_off"], yoff=row["y_off"])
            new_geometries.append(adjusted_geometry)
        return new_geometries

    def _create_expanded_square_bboxes(self, gdf: gpd.GeoDataFrame, square_size: int, image_size: int) -> list:
        """Creates expanded square bounding boxes.

        NOTE: We expand the original bounding boxes to simplify the patching algorithm. Instead of choosing the
        coordinates of the new bounding boxes from the continuous space (step every pixel), we simplify the problem
        focusing on choosing the best bounding boxes from the list of expanded square bounding boxes.

        Args:
                gdf (gpd.GeoDataFrame): A GeoDataFrame containing the bounding boxes.
                square_size (int): The size of the square.
                image_size (int): The size of the image.

        Returns:
            list: A list of expanded square bounding boxes.
        """
        proposed_coordinates = gdf.geometry.apply(lambda x: self._expand_bbox_to_square(x, square_size))
        df_proposed_coordinates = self._calculate_overflow(proposed_coordinates, image_size)
        square_polygons = self._translate_bboxes_to_fit_within_image(df_proposed_coordinates)
        return square_polygons

    def _calculate_score(self, polygon_to_cover: Polygon, covering_box: Polygon) -> float:
        """Calculates the score of the covering box. The score is the area of the intersection minus the area of the
        background.

        NOTE: Throughout the greedy implementation of patching algorithm we are removing the original bboxes that were
        already covered by the new bbox. Therefore, penalization for covering the background accounts also for the
        penalization for overlapping new bboxes. Possible improvement could be to increase the penalization when there
        are more than two overlapping bboxes.

        Args:
            polygon_to_cover (Polygon): The polygon to cover.
            covering_box (Polygon): The covering box.

        Returns:
            float: The score of the covering box.
        """
        intersection = polygon_to_cover.intersection(covering_box)
        background = covering_box.difference(polygon_to_cover)
        return intersection.area - background.area

    def _add_score_to_gdf(self, adjusted_gdf: gpd.GeoDataFrame, merged_polygon: Polygon) -> gpd.GeoDataFrame:
        """Adds a score to the GeoDataFrame.

        Args:
            adjusted_gdf (gpd.GeoDataFrame): The GeoDataFrame to adjust.
            merged_polygon (Polygon): The polygon to merge.

        Returns:
            gpd.GeoDataFrame: The adjusted GeoDataFrame.
        """
        is_covered = adjusted_gdf["is_covered"]
        adjusted_gdf.loc[~is_covered, "score"] = adjusted_gdf.loc[~is_covered, "geometry"].apply(
            lambda x: self._calculate_score(merged_polygon, x)
        )
        return adjusted_gdf

    def _calculate_patch_coverage(self, polygon_to_cover: Polygon, covering_box: Polygon) -> float:
        """Calculates the patch coverage - what percentage of the covering box is covered by the original bounding
        boxes.

        Args:
            polygon_to_cover (Polygon): The polygon to cover.
            covering_box (Polygon): The covering box.

        Returns:
            float: The patch coverage.
        """
        intersection = polygon_to_cover.intersection(covering_box).area
        return intersection / covering_box.area

    def _calculate_global_coverage_percentage(self, polygon_to_cover: Polygon, covered_polygon: Polygon) -> float:
        """Calculates the global coverage percentage - what percentage of the polygon to cover (original bounding boxes)
         is covered by the covered_polygon (new bounding boxes).

        Args:
            polygon_to_cover (Polygon): The polygon to cover.
            covered_polygon (Polygon): The covered polygon.

        Returns:
            float: The global coverage percentage.
        """

        intersection = polygon_to_cover.intersection(covered_polygon).area
        return intersection / polygon_to_cover.area

    def _greedy_patching(
        self, adjusted_gdf: gpd.GeoDataFrame, original_polygon: Polygon, image_size: int, square_size: int
    ) -> gpd.GeoDataFrame:
        """Generates a set of square patches that cover the original polygon, using a greedy algorithm. The algorithm
        iteratively selects the best patch to cover the original polygon. Best patch is the one that covers the largest
        area of the original polygon. The algorithm stops when the original polygon is covered by at least 80% or when
        the area of the patches exceeds the area of the image.

        Args:
            adjusted_gdf (gpd.GeoDataFrame): A GeoDataFrame containing the bounding boxes.
            original_polygon (Polygon): The original polygon within the image.
            image_size (int): The size of the square image's side.
            square_size (int): The size of one side of each square patch.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing the patches as geometries.
        """
        merged_polygon = deepcopy(original_polygon)
        adjusted_gdf["is_covered"] = False
        adjusted_gdf["score"] = 0.0
        global_coverage_percentage = 0.0
        area_of_adjusted_patches = 0
        image_area = image_size**2

        while (
            (global_coverage_percentage < 0.8)
            and (area_of_adjusted_patches <= image_area)
            and (not adjusted_gdf["is_covered"].all())
        ):
            adjusted_gdf = self._add_score_to_gdf(adjusted_gdf, merged_polygon)
            is_covered = adjusted_gdf["is_covered"]
            best_box_id = adjusted_gdf.loc[~is_covered, "score"].idxmax()
            adjusted_gdf.loc[best_box_id, "is_covered"] = True
            covered_polygon = adjusted_gdf.loc[adjusted_gdf["is_covered"]].geometry.unary_union
            merged_polygon = merged_polygon.difference(covered_polygon)
            global_coverage_percentage = original_polygon.intersection(covered_polygon).area / original_polygon.area
            area_of_adjusted_patches = sum(is_covered) * square_size**2

        patched_gdf = adjusted_gdf.loc[adjusted_gdf["is_covered"]].drop(columns=["is_covered", "score"]).copy()
        return patched_gdf

    def _grid_patching(
        self, adjusted_gdf: gpd.GeoDataFrame, original_polygon: Polygon, image_size: int, square_size: int
    ) -> gpd.GeoDataFrame:
        """Generates a grid of square patches within a square image, adjusting the grid to align the patches' centroid
        as closely as possible with the centroid of the original polygon, without exceeding the image boundaries.

        Example:
            X - represents the pixels
            0 - represents the centroid of the original polygon
            # - represents the grid of square patches

            Initial grid:
                # # # # X X
                # # # # X X
                # # # # X X
                # # # # 0 X
                X X X X X X
                X X X X X X

            After adjusting the grid:
                X X X X X X
                X X X X X X
                X X # # # #
                X X # # 0 #
                X X # # # #
                X X # # # #

        Args:
                original_polygon (Polygon): The original polygon within the image.
                image_size (int): The size of the square image's side.
                square_size (int): The size of one side of each square patch.

                Returns:
                        gpd.GeoDataFrame: A GeoDataFrame containing the patches as geometries.
        """
        image_center = image_size / 2

        poly_centroid = original_polygon.centroid
        poly_centroid_x, poly_centroid_y = poly_centroid.x, poly_centroid.y

        displacement_x = round(poly_centroid_x - image_center)
        displacement_y = round(poly_centroid_y - image_center)

        num_patches_per_side = image_size // square_size
        occupied_space = num_patches_per_side * square_size
        offset = (image_size - occupied_space) // 2

        start_x = offset + displacement_x
        start_y = offset + displacement_y

        start_x = max(0, min(start_x, image_size - occupied_space))
        start_y = max(0, min(start_y, image_size - occupied_space))

        x = np.arange(start_x, occupied_space, square_size)
        y = np.arange(start_y, occupied_space, square_size)
        xx, yy = np.meshgrid(x, y)

        patches = []
        for x, y in zip(xx.flatten(), yy.flatten()):
            patches.append(box(x, y, x + square_size, y + square_size))

        patched_gdf = adjusted_gdf.drop(columns=["geometry"]).drop_duplicates()
        assert patched_gdf.shape[0] == 1
        polygons = gpd.GeoDataFrame(geometry=patches)
        patched_gdf = patched_gdf.merge(polygons, how="cross").set_geometry("geometry")
        return patched_gdf

    def _create_geodataframe(self, df_image_data: pd.DataFrame, df_cell_locations: pd.DataFrame) -> gpd.GeoDataFrame:
        """Creates a GeoDataFrame from a DataFrame.

        Args:
            df_image_data (pd.DataFrame): The DataFrame containing image metadata.
            df_cell_locations (pd.DataFrame): The DataFrame containing cell locations.

        Returns:
            gpd.GeoDataFrame: The GeoDataFrame.
        """
        primary_key_columns = [c.ASSAY_PLATE_BARCODE, c.WELL_POSITION, c.FIELD]
        assert df_image_data[primary_key_columns].drop_duplicates().shape[0] == df_image_data.shape[0]

        df_combined = df_cell_locations.merge(
            df_image_data, on=[c.ASSAY_PLATE_BARCODE, c.WELL_POSITION, c.FIELD], how="inner"
        )
        df_combined["geometry"] = df_combined.apply(
            lambda x: box(x[c.MIN_X], x[c.MIN_Y], x[c.MAX_X], x[c.MAX_Y]), axis=1
        )
        df_combined = df_combined.drop(
            columns=[c.MIN_X, c.MIN_Y, c.MAX_X, c.MAX_Y, c.OBJECT_NUMBER, c.IMAGE_NUMBER, c.AREA]
        )
        gdf_combined = gpd.GeoDataFrame(df_combined, geometry="geometry")
        return gdf_combined

    def _star_wrapper(self, kwargs, function: str) -> pd.DataFrame:  # fixme: make me a function decorator
        return self.__getattribute__(function)(**kwargs)

    def _assign_new_patches_for_image(
        self, image_gdf: gpd.GeoDataFrame, square_size: int, image_size: int, minimal_coverage_percentage: float
    ) -> pd.DataFrame:
        image_area = image_size**2
        original_gdf = image_gdf.drop(columns=["adjusted_geometry"]).set_geometry("geometry")
        adjusted_gdf = (
            image_gdf.drop(columns=["geometry"])
            .rename(columns={"adjusted_geometry": "geometry"})
            .set_geometry("geometry")
        )
        original_polygon = original_gdf.geometry.unary_union
        original_bboxes_coverage_percentage = original_polygon.area / image_area
        if original_bboxes_coverage_percentage < minimal_coverage_percentage:
            patched_gdf = self._greedy_patching(adjusted_gdf, original_polygon, image_size, square_size)
        else:
            patched_gdf = self._grid_patching(
                adjusted_gdf=adjusted_gdf,
                original_polygon=original_polygon,
                image_size=image_size,
                square_size=square_size,
            )

        patched_gdf["patch_coverage_percentage"] = patched_gdf.geometry.apply(
            lambda x: self._calculate_patch_coverage(original_polygon, x)
        )
        patched_gdf["global_coverage_percentage"] = self._calculate_global_coverage_percentage(
            original_polygon, patched_gdf.geometry.unary_union
        )
        return patched_gdf

    def _data_iterator(
        self, gdf: gpd.GeoDataFrame, square_size: int, image_size: int, minimal_coverage_percentage: float
    ) -> dict:
        for _, group in gdf.groupby([c.ASSAY_PLATE_BARCODE, c.WELL_POSITION, c.FIELD]):
            yield {
                "image_gdf": group,
                "square_size": square_size,
                "image_size": image_size,
                "minimal_coverage_percentage": minimal_coverage_percentage,
            }

    def _assign_new_patches(
        self, gdf: gpd.GeoDataFrame, square_size: int, image_size: int, minimal_coverage_percentage: float
    ) -> pd.DataFrame:
        dataframes_with_new_patches: list = []
        total_iterations = gdf.groupby([c.ASSAY_PLATE_BARCODE, c.WELL_POSITION, c.FIELD]).ngroups
        data_iterator = self._data_iterator(gdf, square_size, image_size, minimal_coverage_percentage)
        star_assign_new_patches_for_image_function = partial(
            self._star_wrapper, function="_assign_new_patches_for_image"
        )
        with Pool(24) as pool:
            for patched_gdf in tqdm(
                pool.imap_unordered(star_assign_new_patches_for_image_function, data_iterator), total=total_iterations
            ):
                dataframes_with_new_patches.append(patched_gdf)
        df_new_patches = pd.concat(dataframes_with_new_patches)
        new_patches_coordinates: pd.DataFrame = df_new_patches.geometry.bounds
        df_new_patches = pd.concat([df_new_patches.drop(columns=["geometry"]), new_patches_coordinates], axis=1)
        return df_new_patches

    def transform(
        self, df_cell_locations, df_image_metadata, new_patch_size, input_size, minimal_coverage_percentage
    ) -> pd.DataFrame:

        gdf_combined = self._create_geodataframe(df_image_data=df_image_metadata, df_cell_locations=df_cell_locations)
        expanded_original_patches = self._create_expanded_square_bboxes(
            gdf=gdf_combined, square_size=new_patch_size, image_size=input_size
        )
        gdf_combined["adjusted_geometry"] = gpd.GeoSeries(expanded_original_patches)
        df_with_new_patches = self._assign_new_patches(
            gdf=gdf_combined,
            square_size=new_patch_size,
            image_size=input_size,
            minimal_coverage_percentage=minimal_coverage_percentage,
        )
        return df_with_new_patches
