import os
import cv2
import time

from filters.ColorMasking import ColorMaskingFilter
from filters.ContourDetection import ContourDetectionFilter
from filters.GrabCut import ContourDetection2Filter
from filters.KMeans import KMeansFilter
from filters.OtsuThreshold import OtsuThresholdFilter
from filters.MedianFilter import MedianFilter
from filters.BilateralFilter import BilateralFilter
from filters.AnisotropicDiffusionFilter import AnisotropicDiffusionFilter
from filters.AdaptiveGaussianFilter import AdaptiveGaussianFilter
from filters.DownsamplingFilter import DownsamplingFilter
from filters.HoughTransformFilter import HoughTransformFilter
from filters.HistogramBasedSegmentation import HistogramSegmentationFilter
from filters.EdgeDetectionandContours import EdgeDetectionFilter
from filters.SuperpixelSegmentation import SuperpixelSegmentationFilter
from filters.TextureAnalysisSegmentation import TextureAnalysisFilter
from filters.Paper import BoundingBox3DFilter
from filters.YOLOv9 import YOLOv9Filter


class Evaluator:
    def __init__(self, input_image_path, output_dir):
        self.input_image_path = input_image_path
        self.output_dir = output_dir
        self.filters = []

    def add_filter(self, filter):
        self.filters.append(filter)

    def apply_filters(self):
        # Carregar a imagem de entrada
        image = cv2.imread(self.input_image_path)
        
        for filter in self.filters:
            initial_time = time.time()
            result = filter.filter(image)
            end_time = time.time()
            elapsed_time = end_time - initial_time
            filter_name = type(filter).__name__
            print(f"Filter {filter_name} applied in {1000*elapsed_time:.2f} ms")

            # Criar diretório para o filtro
            filter_output_dir = os.path.join(self.output_dir, filter_name)
            os.makedirs(filter_output_dir, exist_ok=True)

            # Salvar a imagem final diretamente no diretório de saída
            result_path = os.path.join(self.output_dir, f'{filter_name}_final_result.jpg')
            cv2.imwrite(result_path, result)

            # Salvar imagens intermediárias no diretório do filtro
            filter.save_intermediate_images(filter_output_dir)

if __name__ == "__main__":
    input_image_path = "data/vehicle.jpg"
    output_dir = "output/out0"

    # Configurações para os filtros
    color_masking_config = {
        "lower_blue": (0, 0, 0),
        "upper_blue": (215, 51, 51)
    }
    
    kmeans_config = {
        "K": 3
    }

    median_filter_config = {
        "ksize": 3
    }

    bilateral_filter_config = {
        "d": 9,
        "sigma_color": 75,
        "sigma_space": 75
    }

    anisotropic_diffusion_config = {
        "sigma_color": 0.05,
        "sigma_spatial": 15,
        "multichannel": True
    }

    adaptive_gaussian_config = {
        "ksize": (5, 5),
        "sigma": 0
    }

    downsampling_config = {
        "scale_factor": 0.5
    }

    yolov9_config = {
        "model_path": "models/yolov9-seg.pt"
    }

    # Inicializar o avaliador
    evaluator = Evaluator(input_image_path, output_dir)

    # Adicionar os filtros com as configurações
    evaluator.add_filter(YOLOv9Filter(config=yolov9_config))
    evaluator.add_filter(ColorMaskingFilter(config=color_masking_config))
    evaluator.add_filter(ContourDetectionFilter(config={}))
    evaluator.add_filter(ContourDetection2Filter(config={}))
    evaluator.add_filter(KMeansFilter(config=kmeans_config))
    evaluator.add_filter(OtsuThresholdFilter(config={}))
    evaluator.add_filter(MedianFilter(config=median_filter_config))
    evaluator.add_filter(BilateralFilter(config=bilateral_filter_config))
    evaluator.add_filter(AnisotropicDiffusionFilter(config=anisotropic_diffusion_config))
    evaluator.add_filter(AdaptiveGaussianFilter(config=adaptive_gaussian_config))
    evaluator.add_filter(DownsamplingFilter(config=downsampling_config))
    evaluator.add_filter(HoughTransformFilter(config={}))
    evaluator.add_filter(HistogramSegmentationFilter(config={}))
    evaluator.add_filter(EdgeDetectionFilter(config={}))
    evaluator.add_filter(SuperpixelSegmentationFilter(config={}))
    evaluator.add_filter(TextureAnalysisFilter(config={}))
    evaluator.add_filter(BoundingBox3DFilter(config={}))

    # Aplicar os filtros
    evaluator.apply_filters()