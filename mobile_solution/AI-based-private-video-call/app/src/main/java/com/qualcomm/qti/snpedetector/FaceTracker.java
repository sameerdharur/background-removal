package com.qualcomm.qti.snpedetector;

import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.core.Core;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Mat;
import org.opencv.core.Rect2d;
import org.opencv.android.Utils;
import org.opencv.tracking.TrackerMIL;

import java.util.ArrayList;


public class FaceTracker {
    public static final String LOGTAG = "SNPEDetector:FaceTraker";

    private TrackerMIL tracker;
    private static int gridHeight;
    private static int gridWidth;
    private static int trackedX;
    private static int trackedY;

    private static boolean hasTrackedCoordinates = false;

    private boolean trackerInitCalled = false;


    public FaceTracker(int width, int height) {
        gridWidth = width;
        gridHeight = height;
        Log.d(LOGTAG, "INFO: Creating Tracker ...");
        tracker = TrackerMIL.create();
        Log.d(LOGTAG, "INFO: Created Tracker done!");
    }


    public static void setTrackedCoordinates(int x, int y) {
        trackedY = y;
        trackedX = x;
        hasTrackedCoordinates = true;
    }

    private static Mat bitmapToMat(Bitmap bitmap) {
        Mat rgba_mat = new Mat();
        Bitmap bmp32 = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Utils.bitmapToMat(bmp32, rgba_mat);
        Mat bw_mat = new Mat();
        Imgproc.cvtColor(rgba_mat, bw_mat, Imgproc.COLOR_RGBA2RGB);
        return bw_mat;
    }

    private static Rect2d boxToRect2d(Box box) {
        double left = Math.round(box.left * gridWidth);
        double top = Math.round(box.top * gridHeight);
        double width = Math.round((box.right-box.left) * gridWidth);
        double height = Math.round((box.bottom-box.top) * gridHeight);
        Rect2d rect = new Rect2d(left, top, width, height);
        return rect;
    }

    // Compute the IOU between the two boxes. (Assuming the origin is top left corner.)
    private static double computeIou(Rect2d a, Rect2d b) {
        double aBottom = a.y + a.height;
        double bBottom = b.y + b.height;
        double aRight = a.x + a.width;
        double bRight = b.x + b.width;
        double yA = Math.max(a.y, b.y);
        double xA = Math.max(a.x, b.x);
        double yB = Math.min(aBottom, bBottom);
        double xB = Math.max(aRight, bRight);
        // Compute the intersection area.
        double interArea = Math.max(0.f, (xB - xA + 1)) * Math.max(0.f, (yB - yA + 1));
        double boxAArea = (aBottom - a.x) * (aRight - a.x);
        double boxBArea = (bBottom - b.y) * (bRight - b.x);
        double iou = interArea / (boxAArea + boxBArea - interArea);
        return iou;
    }

    private static Box getNearestBox(int targetX, int targetY, ArrayList<Box> boxes) {
        double nearestDist = 1e9;
        int nearestDistIdx = 0;
        for (int i=0; i < boxes.size(); i++) {
            Box currBox = boxes.get(i);
            Rect2d currRect = boxToRect2d(currBox);
            int boxCenterX = (int)(currRect.x + currRect.width/2);
            int boxCenterY = (int)(currRect.y + currRect.height/2);
            double dist = Math.sqrt(Math.pow(boxCenterX-targetX,2) + Math.pow(boxCenterY-targetY,2));
            if (dist < nearestDist) {
                nearestDist = dist;
                nearestDistIdx = i;
            }
        }
        Box initBox = boxes.get(nearestDistIdx);
        Log.d(LOGTAG, "getNearestBox: Click(" + String.valueOf(targetX) + ", " + String.valueOf(targetY) +")(" + String.valueOf(initBox.left*300) + ", " + String.valueOf(initBox.top*300) +")" );
        return initBox;
    }

    private boolean initFromBox(Bitmap imageBitmap, Box initBox) {
        tracker = TrackerMIL.create();
        Mat image = bitmapToMat(imageBitmap);
        Rect2d initRect = boxToRect2d(initBox);
        return tracker.init(image, initRect);
    }

    private Rect2d updateState(Bitmap imageBitmap) {
        Mat image = bitmapToMat(imageBitmap);
        Rect2d trackedRect = new Rect2d();
        boolean ok = tracker.update(image, trackedRect);
        return ok ? trackedRect: null;
    }

    public ArrayList<Box> removeTrackedBox(Bitmap imageBitmap, ArrayList<Box> boxes, float thresh) {
        // early exit
        if (boxes == null) {
            return null;
        }

        if (hasTrackedCoordinates) {
            Log.d(LOGTAG, "INFO: Tracker was initialized!");
            Box initBox = getNearestBox(trackedX, trackedY, boxes);
            initFromBox(imageBitmap, initBox);
            hasTrackedCoordinates = false;
        }

        // We update no matter what, if track is lost we need to re-init from boxes
        Rect2d trackedRect = updateState(imageBitmap);
        if (trackedRect == null) {
            Log.e(LOGTAG, "ERROR: Track was lost!");
            for (Box box : boxes) box.is_tracked = false;
            return boxes;
        }
        
        // Get box with highest overlap with tracked box
        double highestIoU = 0;
        int highestIoUIdx = 0;
        for (int i=0; i < boxes.size(); i++) {
            Box box = boxes.get(i);
            box.is_tracked = false;
            if (box.type_score < thresh) {
                continue;
            }
            Rect2d currRect = boxToRect2d(box);
            double currIou = computeIou(currRect, trackedRect);
            if (currIou > highestIoU) {
                highestIoU = currIou;
                highestIoUIdx = i;
            }
        }
        Box trackedBox = boxes.get(highestIoUIdx);
        trackedBox.is_tracked = true;
        if (highestIoU > 0.3) {
            initFromBox(imageBitmap, trackedBox);
            Log.w(LOGTAG, "Found matching IoU, creating new tracker...");
        }
        // return boxes or null
        return boxes;
    }
}
