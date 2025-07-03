#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <opencv2/opencv.hpp>
#pragma comment(lib, "opencv_world4110d.lib") // Versão de acordo com a recomendada no Moodle

extern "C" {
#include "vc.h"
}

int thresholdValue = 80;

void vc_timer() {
	static bool running = false;
	static std::chrono::steady_clock::time_point previousTime = std::chrono::steady_clock::now();
	if (!running) {
		running = true;
	}
	else {
		auto currentTime = std::chrono::steady_clock::now();
		std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(currentTime - previousTime);
		std::cout << "Tempo decorrido: " << time_span.count() << " segundos\n";
		std::cout << "Pressione qualquer tecla para continuar...\n";
		std::cin.get();
	}
}

std::string classificarMoeda(double area, const std::string& video, std::map<std::string, int>& contagemMoedas) {
	if (video == "video1.mp4") {
		if (area >= 2000 && area < 2900) { contagemMoedas["1c"]++; return "1c"; }
		else if (area >= 2900 && area < 3000) { contagemMoedas["2c"]++; return "2c"; }
		else if (area >= 3000 && area < 4000) { contagemMoedas["5c"]++; return "5c"; }
		else if (area >= 4000 && area < 6000) { contagemMoedas["10c"]++; return "10c"; }
		else if (area >= 6000 && area < 9000) { contagemMoedas["20c"]++; return "20c"; }
		else if (area >= 11500 && area < 13000) { contagemMoedas["50c"]++; return "50c"; }
		else if (area >= 9000 && area < 16000) { contagemMoedas["1euro"]++; return "1€"; }
		else if (area >= 16000 && area < 17000) { contagemMoedas["2euro"]++; return "2€"; }
	}
	else if (video == "video2.mp4") {
		if (area >= 2000 && area < 2600) { contagemMoedas["1c"]++; return "1c"; }
		else if (area >= 2600 && area < 2900) { contagemMoedas["2c"]++; return "2c"; }
		else if (area >= 2900 && area <= 3720) { contagemMoedas["5c"]++; return "5c"; }
		else if (area > 3720 && area < 4500) { contagemMoedas["10c"]++; return "10c"; }
		else if (area >= 4500 && area < 7800) { contagemMoedas["20c"]++; return "20c"; }
		else if (area >= 7800 && area < 7930) { contagemMoedas["50c"]++; return "50c"; }
		else if (area >= 7930 && area <12000) { contagemMoedas["1euro"]++; return "1€";	}
		else if (area >=12000 && area < 15000) { contagemMoedas["2euro"]++; return "2€";	}
	}

	return "X";
}


int main() {
	std::vector<cv::Point> moedasDetectadas;
	std::string videofile = "video2.mp4";
	int distanciaMinima = 15;
	if (videofile == "video2.mp4") distanciaMinima = 25;
	cv::VideoCapture capture(videofile);

	if (!capture.isOpened()) {
		std::cerr << "Erro ao abrir vídeo\n";
		return -1;
	}

	int width = (int)capture.get(cv::CAP_PROP_FRAME_WIDTH);
	int height = (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT);

	cv::namedWindow("Moedas", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Segmentada", cv::WINDOW_AUTOSIZE);

	vc_timer();

	std::ofstream output("estatisticas_moedas.csv");
	output << "Tipo,Area,Perimetro,Circularidade,CentroX,CentroY\n";

	cv::Mat frame;
	int key = 0;
	int totalMoedas = 0;
	std::map<std::string, int> contagemMoedas;

	while (key != 'q') {
		capture >> frame;
		if (frame.empty()) break;

		IVC* image = vc_image_new(width, height, 3, 255);
		memcpy(image->data, frame.data, width * height * 3);
		IVC* gray = vc_rgb_to_gray(image);
		IVC* binary = vc_gray_to_binary(gray, thresholdValue);

		cv::Mat segmentada(height, width, CV_8UC1, binary->data);
		segmentada = segmentada.clone();

		vc_image_free(gray);
		vc_image_free(binary);

		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(segmentada, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		std::cout << "Contornos encontrados: " << contours.size() << std::endl;

		for (const auto& contour : contours) {
			double area = 0.0;
			for (size_t i = 0; i < contour.size(); i++) {
				const cv::Point& p1 = contour[i];
				const cv::Point& p2 = contour[(i + 1) % contour.size()];
				area += (p1.x * p2.y) - (p2.x * p1.y);
			}
			area = std::abs(area) / 2.0;

			double perimeter = cv::arcLength(contour, true);
			if (area < 2000) continue;

			double circularity = 4 * CV_PI * area / (perimeter * perimeter);

			if (contour.empty()) continue;

			int minX = frame.cols, minY = frame.rows, maxX = 0, maxY = 0;
			for (const auto& p : contour) {
				if (p.x < minX) minX = p.x;
				if (p.y < minY) minY = p.y;
				if (p.x > maxX) maxX = p.x;
				if (p.y > maxY) maxY = p.y;
			}
			int bboxX = minX;
			int bboxY = minY;
			int bboxWidth = maxX - minX;
			int bboxHeight = maxY - minY;

			int cx = bboxX + bboxWidth / 2;
			int cy = bboxY + bboxHeight / 2;
			cv::Point centro(cx, cy);

			bool duplicada = false;
			for (const auto& m : moedasDetectadas) {
				double dx = m.x - centro.x;
				double dy = m.y - centro.y;
				double dist = sqrt(dx * dx + dy * dy);
				if (dist < distanciaMinima) {
					duplicada = true;
					break;
				}
			}
			if (duplicada) continue;

			moedasDetectadas.push_back(centro);

			draw_rectangle_rgb(image, bboxX, bboxY, bboxWidth, bboxHeight, 0, 255, 0, 2);

			std::string tipo = classificarMoeda(area, videofile, contagemMoedas);

			draw_rectangle_rgb(image, cx - 1, cy - 1, 3, 3, 255, 0, 0, 1);
			cv::putText(frame, tipo, cv::Point(cx - 20, cy - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

			output << tipo << "," << int(area) << "," << int(perimeter) << "," << circularity << "," << cx << "," << cy << "\n";

			if (tipo != "X") totalMoedas++;
		}

		memcpy(frame.data, image->data, width * height * 3);
		vc_image_free(image);
		cv::putText(frame, "Total: " + std::to_string(totalMoedas), cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
		cv::imshow("Moedas", frame);
		cv::imshow("Segmentada", segmentada);

		key = cv::waitKey(1) & 0xFF;
	}

	output.close();

	std::cout << "\n===== Estatísticas Finais =====\n";
	std::cout << "Total de moedas detectadas: " << totalMoedas << "\n";
	for (const auto& m : contagemMoedas)
		std::cout << "Moedas de " << m.first << ": " << m.second << "\n";

	vc_timer();
	capture.release();
	cv::destroyAllWindows();
	return 0;
}