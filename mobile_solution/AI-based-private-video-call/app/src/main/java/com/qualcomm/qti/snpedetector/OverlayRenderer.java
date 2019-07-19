package com.qualcomm.qti.snpedetector;

import android.content.Context;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.BitmapFactory;
import android.graphics.Bitmap;
import android.graphics.Path;
import android.graphics.Rect;
import android.graphics.Region;
import android.support.annotation.Nullable;
import android.support.constraint.ConstraintLayout;
import android.util.ArrayMap;
import android.util.AttributeSet;
import android.view.View;
import android.util.Log;

import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;

import java.lang.Math;

import static java.lang.Math.abs;

public class OverlayRenderer extends View {
    public static final String LOGTAG = "SNPEDetector:OverlayRenderer";
    private boolean mFrameOrientation = false; // 0 -> portrait, 1 -> landscape

    private ReentrantLock mLock = new ReentrantLock();
    private ArrayList<Box> mBoxes = new ArrayList<>();

    private boolean mHasResults;
    private Paint mOutlinePaint = new Paint();
    private Paint mFillPaint = new Paint();
    private Paint mEmojiPaint = new Paint();
    private float mBoxScoreThreshold = 0.4f;

    private boolean mEnablePrivacy = true;
    private boolean mCustomCover = false;

    private int mEmojiSize = 300;
    private String mEmoji = "\uD83D\uDE0A";
    private int flag =0;

    private float selectedX;
    private float selectedY;

    public OverlayRenderer(Context context) {
        super(context);
        init();
    }

    public OverlayRenderer(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public OverlayRenderer(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    public void setGeometryOfTheUnderlyingCameraFeed() {
        // TODO...
    }

    public void setOrientation(boolean o) {
        mFrameOrientation = o;
    }

    public void setEnablePrivacy(boolean o){
        mEnablePrivacy = o;
    }


    public void setCustomCover(boolean o){
        mCustomCover = o;
    }
    public void setNextBoxScoreThreshold(float scoreThreshold) {
        mBoxScoreThreshold = scoreThreshold;
    }

    public void setEmojiSize(int size){
        mEmojiSize = size;
    }

    public void setEmoji(String s){
        mEmoji = s;
    }

    public String getEmoji(){
        return mEmoji;
    }
    public float getBoxScoreThreshold() {
        return mBoxScoreThreshold;
    }

    public void setBoxesFromAnotherThread(ArrayList<Box> nextBoxes) {
        mLock.lock();
        if (nextBoxes == null) {
            mHasResults = false;
            for (Box box : mBoxes)
                box.type_score = 0;
        } else {
            mHasResults = true;
            for (int i = 0; i < nextBoxes.size(); i++) {
                final Box otherBox = nextBoxes.get(i);
                if (i >= mBoxes.size())
                    mBoxes.add(new Box());
                otherBox.copyTo(mBoxes.get(i));
            }
        }
        mLock.unlock();
        postInvalidate();
    }

    private void init() {
        mOutlinePaint.setStyle(Paint.Style.FILL);
     //   mOutlinePaint.setStrokeWidth(2);
        mFillPaint.setColor(Color.BLACK);
        mFillPaint.setStyle(Paint.Style.FILL);
        mEmojiPaint.setTextAlign(Paint.Align.CENTER);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        final int viewWidth = getWidth();
        final int viewHeight = getHeight();
        mEmojiPaint.setTextSize(mEmojiSize);

        // in case there were no results, just draw an X on screen.. totally optional
        if (!mHasResults) {
            mOutlinePaint.setColor(Color.WHITE);
            canvas.drawLine(viewWidth, 0, 0, viewHeight, mOutlinePaint);
            canvas.drawLine(0, 0, viewWidth, viewHeight, mOutlinePaint);
            return;
        }

        final int virtualSize = Math.max(viewHeight, viewHeight);
        final int virtualDx = (virtualSize - viewWidth) / 2;
        final int virtualDy = (virtualSize - viewHeight) / 2;

        Path backpath = new Path();
        backpath.moveTo(0, 0);
        if (mFrameOrientation) {
            backpath.lineTo(0, viewWidth);
            backpath.lineTo(viewHeight, viewWidth);
            backpath.lineTo(viewHeight, 0);
        } else {
            backpath.lineTo(0, viewHeight);
            backpath.lineTo(viewWidth, viewHeight);
            backpath.lineTo(viewWidth, 0);
        }
        backpath.setFillType(Path.FillType.EVEN_ODD);


        mLock.lock();
        flag =0;
        float bl=0, bt=0, br=0, bb=0;
        for (int i = 0; i < mBoxes.size(); i++) {

            final Box box = mBoxes.get(i);
            if (!box.type_name.equals("person")){
                continue;
            }
            // skip rendering below the threshold
            if (box.type_score < mBoxScoreThreshold) {
                continue;
            }

            // compute the final geometry
            if (mFrameOrientation) {
                // LANDSCAPE MODE, FRONT CAMERA COORD
                bt = viewHeight - (virtualSize * box.left - virtualDy);
                bl = viewWidth - (virtualSize * box.top - virtualDx);
                bb = viewHeight - (virtualSize * box.right - virtualDy);
                br = viewWidth - (virtualSize * box.bottom - virtualDx);
            } else {
                // PORTRAIT MODE, FRONT CAMERA COORD
                bl = viewWidth - (virtualSize * box.left - virtualDx);
                bt = virtualSize * box.top - virtualDy;
                br = viewWidth - (virtualSize * box.right - virtualDx);
                bb = virtualSize * box.bottom - virtualDy;
            }
            if (mEnablePrivacy) {
                Path path = new Path();
                //path.setFillType(Path.FillType.EVEN_ODD);
                path.addRect(bl , bt  , br, bb , Path.Direction.CW);
                backpath.addPath(path);
                canvas.drawPath(backpath, mFillPaint);
                canvas.drawPath(path, mOutlinePaint);
                flag =1;
            }
        }
        if(mEnablePrivacy && flag == 0) {
            canvas.drawPath(backpath, mFillPaint);
        }
        mLock.unlock();
    }

    public void changeSelectedCoords(float X, float Y){
        selectedX = X;
        selectedY = Y;
    }
}
