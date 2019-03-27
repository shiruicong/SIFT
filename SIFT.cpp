#include<iostream>
#include<cmath>
#include <opencv2/core.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

typedef vector<Mat> Array;
typedef vector<Array> TwoDArray;

TwoDArray GaussianPyramid;   //������˹�˲��Ľ�����  (5,5)
TwoDArray DoG;               //��ֽ�����   (5,4)
TwoDArray Result;

void buildGaussianPyramid(int sig, int octave, int scale)
{
	int k = 0;
	for (int i = 0; i < octave; i++)
	{
		for (int j = 0; j < scale - 1; j++)
		{
			//�� = ��0* 2��(o+s/S)�η�
			//Mat gausskernel_1 = getGaussianKernel(7, pow(2.0, i + (j / scale)*sig));   // 0
			//Mat gausskernel_2 = getGaussianKernel(7, pow(2.0, i + (j+1 / scale)*sig));
			//Mat gausskernel = gausskernel_1 - gausskernel_2;
			//filter2D(GaussianPyramid[i][j], GaussianPyramid[i][j + 1], GaussianPyramid[i][j].depth(), gausskernel_2);
			//GaussianPyramid[i][j].copyTo(GaussianPyramid[i][j + 1]);
			//filter2D(GaussianPyramid[i][j], DoG[i][j], GaussianPyramid[i][j].depth(), gausskernel);
			GaussianBlur(GaussianPyramid[i][j], GaussianPyramid[i][j + 1], Size(7, 7), pow(2.0, i + (j / scale)*sig));
			DoG[i][j] = GaussianPyramid[i][j] - GaussianPyramid[i][j + 1];
		}	
		if(i < octave - 1)
		{
			pyrDown(GaussianPyramid[i][0], GaussianPyramid[i + 1][0]);
		}
	}
}
	

void findScaleSpaceExtream(int octave, int scale)
{
	for (int i = 0; i < octave; i++)
	{
		for (int j = 1; j < scale - 1; j++)
		{
			Mat image = GaussianPyramid[i][j];
			//imshow("image", image);
			int h = DoG[i][j].rows;
			int w = DoG[i][j].cols;
			Mat temp;
			Mat result = Mat::ones(Size(h-2, w-2), image.type());
			Mat result_max = Mat::ones(Size(h-2, w-2), image.type());
			Mat result_min = Mat::ones(Size(h-2, w-2), image.type());
			
			//��ǰ���һ��
			result_max = DoG[i][j](Range(0, h - 2), Range(0, w - 2)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j](Range(1, h - 1), Range(0, w - 2)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j](Range(2, h ), Range(0, w- 2)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//��ǰ��ڶ���
			result_max &= DoG[i][j](Range(0, h - 2), Range(1, w - 1)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j](Range(2, h), Range(1, w - 1)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//��ǰ�������
			result_max &= DoG[i][j](Range(0, h - 2), Range(2, w )) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j](Range(1, h - 1), Range(2, w )) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j](Range(2, h), Range(2, w )) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//�ϲ��һ��
			result_max &= DoG[i][j + 1](Range(0, h - 2), Range(0, w - 2)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j + 1](Range(1, h - 1), Range(0, w - 2)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j + 1](Range(2, h), Range(0, w - 2)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//�ϲ�ڶ���
			result_max &= DoG[i][j + 1](Range(0, h - 2), Range(1, w - 1)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j + 1](Range(1, h - 1), Range(1, w - 1)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j + 1](Range(2, h), Range(1, w - 1)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//�ϲ������
			result_max = DoG[i][j + 1](Range(0, h - 2), Range(2, w)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j + 1](Range(1, h - 1), Range(2, w)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j + 1](Range(2, h), Range(2, w)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//�²��һ��
			result_max &= DoG[i][j - 1](Range(0, h - 2), Range(0, w - 2)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j - 1](Range(1, h - 1), Range(0, w - 2)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j - 1](Range(2, h), Range(0, w - 2)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//�²�ڶ���
			result_max &= DoG[i][j - 1](Range(0, h - 2), Range(1, w - 1)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j - 1](Range(1, h - 1), Range(1, w - 1)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j - 1](Range(2, h), Range(1, w - 1)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//�²������
			result_max &= DoG[i][j - 1](Range(0, h - 2), Range(2, w)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j - 1](Range(1, h - 1), Range(2, w)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j - 1](Range(2, h), Range(2, w)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));



			//��ǰ���һ��
			result_min = DoG[i][j](Range(0, h - 2), Range(0, w - 2)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j](Range(1, h - 1), Range(0, w - 2)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j](Range(2, h), Range(0, w - 2)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//��ǰ��ڶ���
			result_min &= DoG[i][j](Range(0, h - 2), Range(1, w - 1)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j](Range(2, h), Range(1, w - 1)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//��ǰ�������
			result_min &= DoG[i][j](Range(0, h - 2), Range(2, w)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j](Range(1, h - 1), Range(2, w)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j](Range(2, h), Range(2, w)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//�ϲ��һ��
			result_min &= DoG[i][j + 1](Range(0, h - 2), Range(0, w - 2)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j + 1](Range(1, h - 1), Range(0, w - 2)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j + 1](Range(2, h), Range(0, w - 2)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//�ϲ�ڶ���
			result_min &= DoG[i][j + 1](Range(0, h - 2), Range(1, w - 1)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j + 1](Range(1, h - 1), Range(1, w - 1)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j + 1](Range(2, h), Range(1, w - 1)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//�ϲ������
			result_min &= DoG[i][j + 1](Range(0, h - 2), Range(2, w)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j + 1](Range(1, h - 1), Range(2, w)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j + 1](Range(2, h), Range(2, w)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//�²��һ��
			result_min &= DoG[i][j - 1](Range(0, h - 2), Range(0, w - 2)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j - 1](Range(1, h - 1), Range(0, w - 2)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j - 1](Range(2, h), Range(0, w - 2)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//�²�ڶ���
			result_min &= DoG[i][j - 1](Range(0, h - 2), Range(1, w - 1)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j - 1](Range(1, h - 1), Range(1, w - 1)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j - 1](Range(2, h), Range(1, w - 1)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//�²������
			result_min &= DoG[i][j - 1](Range(0, h - 2), Range(2, w)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j - 1](Range(1, h - 1), Range(2, w)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j - 1](Range(2, h), Range(2, w)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));


			result = result_max | result_min;
			result.copyTo(temp);
			int count = 0;
			for (int m = 0; m < temp.rows; m++)
			{
				for (int n = 0; n < temp.cols; n++)
				{
					//cout << temp.at<uchar>(m, n);
					
					if (temp.at<uchar>(m, n))
					{
						count++;
						
						circle(image, Point(m + 1, n + 1), 5, Scalar(255, 0, 0));
					}
				}
					
			}
			imshow("image", image);
			waitKey(0);
			//printf("count----%d", count);
			Result[i][j-1] = temp;
		}
	}
}
			
void init_var(int octave, int scale)
{
	for (int i = 0; i < octave; i++)//ceng
	{
		GaussianPyramid.push_back(Array(scale + 3));  //meiceng
		DoG.push_back(Array(scale + 2));      //
		Result.push_back(Array(scale));
	}
}

	
/*
�´��½�opencv���̵�ʱ��includeĿ¼��LibĿ¼��Ȼ��̳�֮ǰ�����ò�Ӧ�����й���
��Linker -> Input -> additional lib..   �����ڼ̳�֮ǰ�����ã���Ҫ�ֶ����opencv_world401.lib��releaseģʽ�£�opencv_world401d.lib��debugģʽ��
���������libͬʱ��ӵ�releaseģʽ��debugģʽ�£���Ĭ��ѡ�õ�һ��lib������release�����û���debug������
*/
//void mian()
void main()
{
	Mat img, grayImage,image,image_up,image_src;
	img = imread("wall.jpg");
	cvtColor(img, grayImage, COLOR_RGB2GRAY, 0);
	//��˹ƽ��
	//Mat gauss_kernel = getGaussianKernel(5, 1);
	//filter2D(grayImage, image, grayImage.depth(), gauss_kernel);
	GaussianBlur(grayImage, image, Size(7,7), 1);
	//�ϲ���
	pyrUp(image, image_up, Size(image.cols * 2, image.rows * 2));
	GaussianBlur(image_up, image_src, Size(7, 7), 1);
	init_var(5, 2);
	GaussianPyramid[0][0] = image_src;
	buildGaussianPyramid(1,5,2+3);
	//imshow("4-4", DoG[1][0]);
	findScaleSpaceExtream(5, 2+2);
	waitKey(0);
}