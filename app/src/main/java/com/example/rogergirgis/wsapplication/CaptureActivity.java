package com.example.rogergirgis.wsapplication;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.util.Size;
import android.util.TypedValue;
import android.view.WindowManager;

import com.example.rogergirgis.wsapplication.OverlayView.DrawCallback;
import com.example.rogergirgis.wsapplication.env.BorderedText;
import com.example.rogergirgis.wsapplication.env.ImageUtils;
import com.example.rogergirgis.wsapplication.env.Logger;

import java.io.File;
import java.util.Vector;

public class CaptureActivity extends CameraActivity implements OnImageAvailableListener {

    private static final Logger LOGGER = new Logger();

    protected static final boolean SAVE_PREVIEW_BITMAP = true;

    private ResultsView resultsView;

    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private long lastProcessingTimeMs;

    private BorderedText borderedText;
    private Integer sensorOrientation;
    private Classifier classifier;
    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private static final boolean MAINTAIN_ASPECT = true;

    private static final Size DESIRED_PREVIEW_SIZE = new Size(480, 640);
    private static final float TEXT_SIZE_DIP = 10;

    private static final int INPUT_SIZE = 224;
    private static final int IMAGE_MEAN = 117;
    private static final float IMAGE_STD = 1;

    private static final String INPUT_NAME = "input_1";
    private static final String OUTPUT_NAME = "loss/Softmax";
    private static final String MODEL_FILE = "file:///android_asset/Mobilenet_8.pb";
    private static final String LABEL_FILE = "file:///android_asset/classes.8.txt";
    int mPrediction = 0;

    // Saving information
    public static final String FOLDER_KEY = "0";
    public String folderNumber = "0";
    public static final int defaultFolderNumber = 0;
    int IMG_NUMBER = 0;

    Handler handler = new Handler();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        loadConfiguration();

        final String root = Environment.getExternalStorageDirectory().getAbsolutePath() +
                File.separator + "ContinuousImageCaptureApp" + File.separator + folderNumber;
        final File myDir = new File(root);

        if (myDir.isDirectory()) {
            String[] children = myDir.list();
            for (String aChildren : children) {
                new File(myDir, aChildren).delete();
            }
        }
        if (!myDir.mkdirs()) {
            LOGGER.i("Make dir failed");
        }

    }

    private void loadConfiguration() {
        Bundle configuration = getIntent().getExtras();
        folderNumber = Integer.toString(configuration.getInt(FOLDER_KEY, defaultFolderNumber));
    }

    @Override
    public synchronized void onStop() {
        super.onStop();
    }

    @Override
    public void onResume() {
        // Ideally a game should implement onResume() and onPause()
        // to take appropriate action when the activity looses focus
        super.onResume();
    }

    @Override
    public void onPause() {
        // Ideally a game should implement onResume() and onPause()
        // to take appropriate action when the activity looses focus
        super.onPause();
    }

    @Override
    protected void processImage() {
        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

//        runInBackground( new Runnable() {
//                    @Override
//                    public void run() {
//                        // Save IMU data in a csv file with the accompanying image
//                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
//                        requestRender();
//                        readyForNextImage();
//                        ImageUtils.saveBitmap(croppedBitmap, IMG_NUMBER, folderNumber);
//                        IMG_NUMBER++;
//                    }
//        });
        Runnable r = new Runnable() {
            public void run() {
                cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                requestRender();
                readyForNextImage();
                ImageUtils.saveBitmap(croppedBitmap, IMG_NUMBER, folderNumber);
                IMG_NUMBER++;
            }
        };
        handler.postDelayed(r, 400);
    }

    @Override
    protected void onPreviewSizeChosen(Size size, int rotation) {
        final float textSizePx = TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        classifier =
                TensorFlowImageClassifier.create(
                        getAssets(),
                        MODEL_FILE,
                        LABEL_FILE,
                        INPUT_SIZE,
                        IMAGE_MEAN,
                        IMAGE_STD,
                        INPUT_NAME,
                        OUTPUT_NAME);

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = 180; // rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);  //INPUT_SIZE

        frameToCropTransform = ImageUtils.getTransformationMatrix(
                previewWidth, previewHeight,
                previewWidth, previewHeight, // INPUT_SIZE
                sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        renderDebug(canvas);
                    }
                });
    }

    @Override
    protected int getLayoutId() {
        return R.layout.camera_connection_fragment;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    private void renderDebug(final Canvas canvas) {
        if (!isDebug()) {
            return;
        }
        final Bitmap copy = cropCopyBitmap;
        if (copy != null) {
            final Matrix matrix = new Matrix();
            final float scaleFactor = 2;
            matrix.postScale(scaleFactor, scaleFactor);
            matrix.postTranslate(
                    canvas.getWidth() - copy.getWidth() * scaleFactor,
                    canvas.getHeight() - copy.getHeight() * scaleFactor);
            canvas.drawBitmap(copy, matrix, new Paint());

            final Vector<String> lines = new Vector<String>();
            if (classifier != null) {
                String statString = classifier.getStatString();
                String[] statLines = statString.split("\n");
                for (String line : statLines) {
                    lines.add(line);
                }
            }

            lines.add("Frame: " + previewWidth + "x" + previewHeight);
            lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
            lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
            lines.add("Rotation: " + sensorOrientation);
            lines.add("Inference time: " + lastProcessingTimeMs + "ms");
            lines.add("Aggr. Action: " + mPrediction);

            borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
        }
    }

    @Override
    public void onSetDebug(boolean debug) {
        classifier.enableStatLogging(debug);
    }

}
