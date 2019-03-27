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

TwoDArray GaussianPyramid;   //经过高斯滤波的金字塔  (5,5)
TwoDArray DoG;               //差分金字塔   (5,4)
TwoDArray Result;

void buildGaussianPyramid(int sig, int octave, int scale)
{
	int k = 0;
	for (int i = 0; i < octave; i++)
	{
		for (int j = 0; j < scale - 1; j++)
		{
			//σ = σ0* 2的(o+s/S)次方
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
			
			//当前层第一列
			result_max = DoG[i][j](Range(0, h - 2), Range(0, w - 2)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j](Range(1, h - 1), Range(0, w - 2)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j](Range(2, h ), Range(0, w- 2)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//当前层第二列
			result_max &= DoG[i][j](Range(0, h - 2), Range(1, w - 1)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j](Range(2, h), Range(1, w - 1)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//当前层第三列
			result_max &= DoG[i][j](Range(0, h - 2), Range(2, w )) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j](Range(1, h - 1), Range(2, w )) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j](Range(2, h), Range(2, w )) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//上层第一列
			result_max &= DoG[i][j + 1](Range(0, h - 2), Range(0, w - 2)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j + 1](Range(1, h - 1), Range(0, w - 2)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j + 1](Range(2, h), Range(0, w - 2)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//上层第二列
			result_max &= DoG[i][j + 1](Range(0, h - 2), Range(1, w - 1)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j + 1](Range(1, h - 1), Range(1, w - 1)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j + 1](Range(2, h), Range(1, w - 1)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//上层第三列
			result_max = DoG[i][j + 1](Range(0, h - 2), Range(2, w)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j + 1](Range(1, h - 1), Range(2, w)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j + 1](Range(2, h), Range(2, w)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//下层第一列
			result_max &= DoG[i][j - 1](Range(0, h - 2), Range(0, w - 2)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j - 1](Range(1, h - 1), Range(0, w - 2)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j - 1](Range(2, h), Range(0, w - 2)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//下层第二列
			result_max &= DoG[i][j - 1](Range(0, h - 2), Range(1, w - 1)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j - 1](Range(1, h - 1), Range(1, w - 1)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j - 1](Range(2, h), Range(1, w - 1)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//下层第三列
			result_max &= DoG[i][j - 1](Range(0, h - 2), Range(2, w)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j - 1](Range(1, h - 1), Range(2, w)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_max &= DoG[i][j - 1](Range(2, h), Range(2, w)) > DoG[i][j](Range(1, h - 1), Range(1, w - 1));



			//当前层第一列
			result_min = DoG[i][j](Range(0, h - 2), Range(0, w - 2)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j](Range(1, h - 1), Range(0, w - 2)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j](Range(2, h), Range(0, w - 2)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//当前层第二列
			result_min &= DoG[i][j](Range(0, h - 2), Range(1, w - 1)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j](Range(2, h), Range(1, w - 1)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//当前层第三列
			result_min &= DoG[i][j](Range(0, h - 2), Range(2, w)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j](Range(1, h - 1), Range(2, w)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j](Range(2, h), Range(2, w)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//上层第一列
			result_min &= DoG[i][j + 1](Range(0, h - 2), Range(0, w - 2)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j + 1](Range(1, h - 1), Range(0, w - 2)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j + 1](Range(2, h), Range(0, w - 2)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//上层第二列
			result_min &= DoG[i][j + 1](Range(0, h - 2), Range(1, w - 1)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j + 1](Range(1, h - 1), Range(1, w - 1)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j + 1](Range(2, h), Range(1, w - 1)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//上层第三列
			result_min &= DoG[i][j + 1](Range(0, h - 2), Range(2, w)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j + 1](Range(1, h - 1), Range(2, w)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j + 1](Range(2, h), Range(2, w)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//下层第一列
			result_min &= DoG[i][j - 1](Range(0, h - 2), Range(0, w - 2)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j - 1](Range(1, h - 1), Range(0, w - 2)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j - 1](Range(2, h), Range(0, w - 2)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//下层第二列
			result_min &= DoG[i][j - 1](Range(0, h - 2), Range(1, w - 1)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j - 1](Range(1, h - 1), Range(1, w - 1)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));
			result_min &= DoG[i][j - 1](Range(2, h), Range(1, w - 1)) < DoG[i][j](Range(1, h - 1), Range(1, w - 1));

			//下层第三列
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
下次新建opencv工程的时候，include目录和Lib目录仍然会继承之前的配置并应用所有工程
在Linker -> Input -> additional lib..   不会在继承之前的配置，需要手动添加opencv_world401.lib到release模式下，opencv_world401d.lib到debug模式下
如果把两个lib同时添加到release模式和debug模式下，会默认选用第一个lib，导致release不能用或者debug不能用
*/
//void mian()
void main()
{
	Mat img, grayImage,image,image_up,image_src;
	img = imread("wall.jpg");
	cvtColor(img, grayImage, COLOR_RGB2GRAY, 0);
	//高斯平滑
	//Mat gauss_kernel = getGaussianKernel(5, 1);
	//filter2D(grayImage, image, grayImage.depth(), gauss_kernel);
	GaussianBlur(grayImage, image, Size(7,7), 1);
	//上采样
	pyrUp(image, image_up, Size(image.cols * 2, image.rows * 2));
	GaussianBlur(image_up, image_src, Size(7, 7), 1);
	init_var(5, 2);
	GaussianPyramid[0][0] = image_src;
	buildGaussianPyramid(1,5,2+3);
	//imshow("4-4", DoG[1][0]);
	findScaleSpaceExtream(5, 2+2);
	waitKey(0);
}