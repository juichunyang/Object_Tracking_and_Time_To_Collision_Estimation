
#include <iostream>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <ctime>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;

        // pixel coordinates
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        }

        if (enclosingBoxes.size() == 1)
        {
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    }
}

/*
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size.
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); it1++)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        int top=1e8, left=1e8, bottom=0.0, right=0.0;
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); it2++)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; i++)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cout << "Press key to continue to next image" << endl;
        cv::waitKey(0);
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, vector<cv::KeyPoint> &kptsPrev, vector<cv::KeyPoint> &kptsCurr, vector<cv::DMatch> &kptMatches)
{
    boundingBox.keypoints.clear();
    boundingBox.kptMatches.clear();

    std::vector<int> matchIdx;
    float distMean = 0.0;

    for (int idx = 0; idx < kptMatches.size(); idx++)
    {
        // match.trainIdx : train descriptor index
        // match.queryIdx : query descriptor index
        cv::DMatch match = kptMatches[idx];
        cv::Point2f currPt = kptsCurr[match.trainIdx].pt;
        cv::Point2f prevPt = kptsPrev[match.queryIdx].pt;

        if (boundingBox.roi.contains(currPt))
        {
            matchIdx.push_back(idx);
            distMean += cv::norm(currPt - prevPt);
        }

    }
    distMean /= matchIdx.size();

    for (int idx : matchIdx)
    {
        cv::DMatch match = kptMatches[idx];
        cv::Point2f currPt = kptsCurr[match.trainIdx].pt;
        cv::Point2f prevPt = kptsPrev[match.queryIdx].pt;
        double dist = cv::norm(currPt - prevPt);

        if (dist <= distMean * 1.2)
        {
            boundingBox.kptMatches.push_back(match);
            boundingBox.keypoints.push_back(kptsCurr[match.trainIdx]);
        }
    }
}


// Calculate the median
double median(vector<double>& vec)
{
    sort(vec.begin(), vec.end());
    long midIdx = floor(vec.size() / 2);
    double medianNum = vec.size() % 2 == 0 ? (vec[midIdx] + vec[midIdx - 1]) / 2.0 : vec[midIdx];
    return medianNum;
}


// Identify outliers by IQR
void checkOutliers(vector<double>& nums)
{
    // If there are only three numbers, IQR cannot identify any outliers.
    if (nums.size() <= 3)
        return;

    sort(nums.begin(), nums.end());

    // find Q1
    long leftIdx = floor(nums.size() / 2);
    vector<double> q1Vec(nums.begin(), nums.begin() + leftIdx);
    double q1 = median(q1Vec);

    // find Q3
    long rightIdx = nums.size() % 2 == 0 ? leftIdx : leftIdx + 1;
    vector<double> q3Vec(nums.begin() + leftIdx, nums.end());
    double q3 = median(q3Vec);

    // calculate IQR
    double iqr = q3 - q1;

    for (int idx = 0; idx < nums.size(); idx++)
    {
        // remove outliers
        if (nums[idx] < (q1 - 1.5 * iqr) || nums[idx] > (q3 + 1.5 * iqr))
        {
              nums.erase(nums.begin() + idx);
            idx--;
        }

    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(vector<cv::KeyPoint> &kptsPrev, vector<cv::KeyPoint> &kptsCurr, vector<cv::DMatch> kptMatches, double frameRate, double &TTC, const cv::Mat &visImgPrev, const cv::Mat &visImgCurr)
{
    vector<double> distRatio;
    double minDist = 100.0;    // min. required distance

    // compute distance ratio of all matched keypoints
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end(); it1++)
    {
        cv::Point2f kpOuterCurr = kptsCurr[it1->trainIdx].pt;
        cv::Point2f kpOuterPrev = kptsPrev[it1->queryIdx].pt;

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); it2++)
        {
            cv::Point2f kpInnerCurr = kptsCurr[it2->trainIdx].pt;
            cv::Point2f kpInnerPrev = kptsPrev[it2->queryIdx].pt;

            double distCurr = cv::norm(kpOuterCurr - kpInnerCurr);
            double distPrev = cv::norm(kpOuterPrev - kpInnerPrev);

            // avoid division by zero
            if (distPrev > numeric_limits<double>::epsilon() && distCurr >= minDist)
            {
                double ratio = distCurr / distPrev;
                distRatio.push_back(ratio);
            }
        }
    }

    // only continue if list of distance ratios is not empty
    if (distRatio.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // compute camera-based TTC from distance ratios
    double dT = 1 / frameRate;
    checkOutliers(distRatio);
    double medianDistRatio =  median(distRatio);
    TTC = - dT / (1 - medianDistRatio);

    // draw matches
    bool bVis = false;
    if (bVis)
    {
        if (!visImgPrev.empty() && !visImgCurr.empty())
        {
            cv::Mat matchImg;
            cv::drawMatches(visImgPrev, kptsPrev, visImgCurr, kptsCurr, kptMatches, matchImg, cv::Scalar::all(-1), cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            string windowName = "Matching keypoints between current and previous frame";
            cv::namedWindow(windowName, 5);
            cv::imshow(windowName, matchImg);
            cout << "Press key to continue to next image" << endl;
            cv::waitKey(0);  // wait for key to be pressed
        }
    }

}

// Identify outliers by RANSAC
std::unordered_set<int> Ransac3D(std::vector<LidarPoint> &lidarPoints, int maxIterations, float distanceTol)
{
	// Time segmentation process
	//auto startTime = std::chrono::steady_clock::now();

	std::unordered_set<int> inliersResult;
	srand(time(NULL));
	int cloudSize = lidarPoints.size();

	while (maxIterations--)
  {

		std::unordered_set<int> inliers;

		while (inliers.size() < 3)
			inliers.insert(rand() % cloudSize);

		float x1, y1, z1, x2, y2, z2, x3, y3, z3;
		auto itr = inliers.begin();
		x1 = lidarPoints[*itr].x;
		y1 = lidarPoints[*itr].y;
		z1 = lidarPoints[*itr].z;
		itr++;
		x2 = lidarPoints[*itr].x;
		y2 = lidarPoints[*itr].y;
		z2 = lidarPoints[*itr].z;
		itr++;
		x3 = lidarPoints[*itr].x;
		y3 = lidarPoints[*itr].y;
		z3 = lidarPoints[*itr].z;

		float a, b, c, d;
		a = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1);
		b = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1);
		c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
		d = -(a * x1 + b * y1 + c * z1);

		for (int idx = 0; idx < cloudSize; idx++)
    {

			if (inliers.count(idx) > 0)
        continue;

			float x = lidarPoints[idx].x;
			float y = lidarPoints[idx].y;
			float z = lidarPoints[idx].z;
			float dist = fabs(a * x + b * y + c * z + d) / sqrt(a * a + b * b + c * c);

			if (dist <= distanceTol)
				inliers.insert(idx);

		}

		if (inliers.size() > inliersResult.size())
			inliersResult = inliers;
	}

	//auto endTime = std::chrono::steady_clock::now();
	//auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
	//std::cout << "RANSAC took " << elapsedTime.count() << " milliseconds." << std::endl;
  //std::cout << "RANSAC done" << std::endl;

	return inliersResult;

}


// Compute time-to-collision (TTC) based on LiDAR in successive images
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev, std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double dT = 1 / frameRate;  // time between two measurements in seconds
    double laneWidth = 4.0;  // aussumed width of the ego lane

    // find outliers by RANSAC
    int maxIterations = 150;
    float distanceTol = 0.08;
    std::unordered_set<int> prevInliers = Ransac3D(lidarPointsPrev, maxIterations, distanceTol);
    std::unordered_set<int> currInliers = Ransac3D(lidarPointsCurr, maxIterations, distanceTol);

    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;
    for (auto it = prevInliers.begin(); it != prevInliers.end(); it++)
    {
        LidarPoint p = lidarPointsPrev[*it];
        // 3D point within ego lane?
        if (fabs(p.y) <= laneWidth / 2)
        {
            minXPrev = minXPrev > p.x ? p.x : minXPrev;
        }
    }

    for (auto it = currInliers.begin(); it != currInliers.end(); it++)
    {
        LidarPoint p = lidarPointsCurr[*it];
        // 3D point within ego lane?
        if (fabs(p.y) <= laneWidth / 2)
        {
            minXCurr = minXCurr > p.x ? p.x : minXCurr;
        }
    }

    // only continue if list of inliers is not empty
    if (prevInliers.size() == 0 || currInliers.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev - minXCurr);

}


// Match bouding boxes in successive images
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    map<int,int> kptCount;
    int preBBSize = prevFrame.boundingBoxes.size();
    int currBBSize = currFrame.boundingBoxes.size();

    for (auto match : matches)
    {
        // match.trainIdx : train descriptor index
        // match.queryIdx : query descriptor index
        cv::Point2f currPt = currFrame.keypoints[match.trainIdx].pt;
        cv::Point2f prevPt = prevFrame.keypoints[match.queryIdx].pt;

        for (auto prevBox : prevFrame.boundingBoxes)
        {
            for (auto currBox : currFrame.boundingBoxes)
            {
                if (prevBox.roi.contains(prevPt) && currBox.roi.contains(currPt))
                {
                    int key = prevBox.boxID * currBBSize + currBox.boxID;
                    kptCount[key] += 1;
                }
            }
        }
    }

    // Use a vector to record the best match which has the highest number of keypoints correspondences.
    // This method works is simply becuase the naming system of bounding box ID is based on the size of the bouding box vector.
    //  previous Bounding Box 1   previous Bounding Box 2 ...
    // {{currBoxID, maxPoints}, {currBoxID, maxPoints},.........}
    vector<vector<int>> maxCount(preBBSize, vector<int>(2,-1));
    for (auto it : kptCount)
    {
        int preBoxID = it.first / currBBSize;
        int currBoxID = it.first % currBBSize;
        int numPt = it.second;
        if (maxCount[preBoxID][1] < numPt)
        {
            maxCount[preBoxID][1] = numPt;
            maxCount[preBoxID][0] = currBoxID;
        }
    }

    // Put the result to the map
    for (int idx = 0 ; idx < maxCount.size(); idx++)
    {
        bbBestMatches[idx] = maxCount[idx][0];
    }
}
