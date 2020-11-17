package com.gev.hw2;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.content.ClipData;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageDecoder;
import android.graphics.Paint;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;


import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import com.gev.hw2.mtcnn.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

public class MainActivity extends AppCompatActivity {
    private final int REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS = 124;
    public final static int PICK_PHOTO_CODE = 1046;
    private ArrayList<Uri> mArrayUri;
    private ArrayList<Bitmap> mBitmapsSelected;
    private MTCNNModel mtcnnFaceDetector=null;
    private static final String TAG = "Face Clustering";
    private static int minFaceSize=40;
    private AgeGenderEthnicityTfLiteClassifier facialAttributeClassifier=null;
    ArrayList<List<FaceFeatures>> result_feats=null;
    HashMap<Uri, Vector<Box>> boxmap;
    HashMap<Uri, Bitmap> imagemap;
    private Module module;
    int num_clusters = 0;
    ArrayList<Uri> clusterized_uri = new ArrayList<>();
    ArrayList<Integer> clusterized_nums = new ArrayList<>();
    int current_num = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(this, getRequiredPermissions(), REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS);
        }
        else {
            try {
                init();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        TextView textView = (TextView) findViewById(R.id.text);
        ImageView imageView = (ImageView) findViewById(R.id.photoView);
        Button startButton = (Button)findViewById(R.id.start_button);
        startButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                intent.setType("image/*");
                intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
                startActivityForResult(Intent.createChooser(intent, "Select Picture"), 1000);
            }
        });

        Button clustButton = (Button)findViewById(R.id.clusterize_button);
        clustButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (mBitmapsSelected.size() > 0) {
                    current_num = 0;
                    Spinner spinner = findViewById(R.id.spinner);
                    int selected = spinner.getSelectedItemPosition();

                    textView.setText("");
                    if (selected == 0) {
                        HashMap<FaceFeatures, Uri> featsMap = new HashMap<>();
                        for (int i = 0; i < mBitmapsSelected.size(); ++i) {
                            Bitmap bitmap = mBitmapsSelected.get(i);
                            Uri uri = mArrayUri.get(i);
                            List<FaceFeatures> features = getFacesFeatures(bitmap);
                            for (FaceFeatures feat : features) {
                                featsMap.put(feat, uri);
                            }
                        }
                        if (featsMap.size() > 1) {
                            ArrayList<FaceFeatures> keyList = new ArrayList<>(featsMap.keySet());
                            try {
                                DBSCANClusterer<FaceFeatures> clusterer = new DBSCANClusterer<>(
                                        keyList, 2, 1, new DistanceMetricFaceFeatures()
                                );
                                ArrayList<ArrayList<FaceFeatures>> clusters = clusterer.performClustering();

                                int i = 1;
                                num_clusters = 0;
                                clusterized_uri.clear();
                                clusterized_nums.clear();

                                for (ArrayList<FaceFeatures> feature : clusters) {
                                    for (FaceFeatures face : feature) {
                                        String name = featsMap.get(face).getPath();
                                        featsMap.remove(face);
                                        clusterized_uri.add(featsMap.get(face));
                                        clusterized_nums.add(num_clusters);
                                    }
                                    num_clusters++;

                                }

                                if (featsMap.size() > 0) {
                                    for (FaceFeatures face : featsMap.keySet()) {
                                        String name = featsMap.get(face).getPath();
                                        clusterized_uri.add(featsMap.get(face));
                                        clusterized_nums.add(num_clusters);
                                        num_clusters++;
                                    }
                                }
                            } catch (DBSCANClusteringException e) {
                                e.printStackTrace();
                            }
                        }
                    } else {
                        HashMap<float[], Uri> featsMap = new HashMap<>();
                        for (int i = 0; i < mBitmapsSelected.size(); ++i) {
                            Bitmap bitmap = mBitmapsSelected.get(i);
                            Uri uri = mArrayUri.get(i);
                            Bitmap bmp = bitmap;
                            Bitmap resizedBitmap = bmp;
                            double minSize = 600.0;
                            double scale = Math.min(bmp.getWidth(), bmp.getHeight()) / minSize;
                            if (scale > 1.0) {
                                resizedBitmap = Bitmap.createScaledBitmap(bmp, (int) (bmp.getWidth() / scale), (int) (bmp.getHeight() / scale), false);
                                bmp = resizedBitmap;
                            }
                            List<float[]> facesInfo = new ArrayList<>();
                            try {
                                Vector<Box> bboxes = boxmap.get(uri);

                                for (Box box : bboxes) {
                                    android.graphics.Rect bbox = new android.graphics.Rect(bmp.getWidth() * box.left() / resizedBitmap.getWidth(),
                                            bmp.getHeight() * box.top() / resizedBitmap.getHeight(),
                                            bmp.getWidth() * box.right() / resizedBitmap.getWidth(),
                                            bmp.getHeight() * box.bottom() / resizedBitmap.getHeight()
                                    );
                                    Bitmap faceBitmap = Bitmap.createBitmap(bmp, bbox.left, bbox.top, bbox.width(), bbox.height());
                                    Bitmap resultBitmap = Bitmap.createScaledBitmap(faceBitmap, 224, 224, false);
                                    final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resultBitmap,
                                            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
                                    Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

                                    float[] scores = outputTensor.getDataAsFloatArray();
                                    facesInfo.add(scores);
                                }
                            } catch (Exception e) {
                                continue;
                            }
                            for (float[] feat : facesInfo) {
                                featsMap.put(feat, uri);
                            }
                        }

                        if (featsMap.size() > 1) {
                            ArrayList<float[]> keyList = new ArrayList<>(featsMap.keySet());
                            try {
                                DBSCANClusterer<float[]> clusterer = new DBSCANClusterer<>(
                                        keyList, 2, 1, new DistanceMetricFloat()
                                );
                                ArrayList<ArrayList<float[]>> clusters = clusterer.performClustering();

                                int i = 1;
                                num_clusters = 0;
                                clusterized_uri.clear();
                                clusterized_nums.clear();

                                for (ArrayList<float[]> feature : clusters) {
                                    for (float[] face : feature) {
                                        String name = featsMap.get(face).getPath();
                                        featsMap.remove(face);
                                        clusterized_uri.add(featsMap.get(face));
                                        clusterized_nums.add(num_clusters);
                                    }
                                    num_clusters++;
                                }
                                if (featsMap.size() > 0) {
                                    for (float[] face : featsMap.keySet()) {
                                        String name = featsMap.get(face).getPath();
                                        clusterized_uri.add(featsMap.get(face));
                                        clusterized_nums.add(num_clusters);
                                        num_clusters++;
                                    }
                                }
                            } catch (DBSCANClusteringException e) {
                                e.printStackTrace();
                            }
                        }
                    }
                    Uri imgUri = clusterized_uri.get(current_num);
                    Bitmap bmp = imagemap.get(imgUri);

                    Bitmap resizedBitmap = bmp;
                    double minSize = 600.0;
                    double scale = Math.min(bmp.getWidth(), bmp.getHeight()) / minSize;
                    if (scale > 1.0) {
                        resizedBitmap = Bitmap.createScaledBitmap(bmp, (int) (bmp.getWidth() / scale), (int) (bmp.getHeight() / scale), false);
                        bmp = resizedBitmap;
                    }
                    Vector<Box> bboxes = mtcnnFaceDetector.detectFaces(resizedBitmap, minFaceSize);//(int)(bmp.getWidth()*MIN_FACE_SIZE));

                    Bitmap tempBmp = Bitmap.createBitmap(bmp.getWidth(), bmp.getHeight(), Bitmap.Config.ARGB_8888);
                    Canvas c = new Canvas(tempBmp);
                    Paint p = new Paint();
                    p.setStyle(Paint.Style.STROKE);
                    p.setAntiAlias(true);
                    p.setFilterBitmap(true);
                    p.setDither(true);
                    p.setColor(Color.BLUE);
                    p.setStrokeWidth(5);

                    Paint p_text = new Paint();
                    p_text.setColor(Color.WHITE);
                    p_text.setStyle(Paint.Style.FILL);
                    p_text.setColor(Color.GREEN);
                    p_text.setTextSize(24);

                    c.drawBitmap(bmp, 0, 0, null);

                    for (Box box : bboxes) {

                        p.setColor(Color.RED);
                        android.graphics.Rect bbox = new android.graphics.Rect(bmp.getWidth() * box.left() / resizedBitmap.getWidth(),
                                bmp.getHeight() * box.top() / resizedBitmap.getHeight(),
                                bmp.getWidth() * box.right() / resizedBitmap.getWidth(),
                                bmp.getHeight() * box.bottom() / resizedBitmap.getHeight()
                        );

                        c.drawRect(bbox, p);

                        if (facialAttributeClassifier != null) {
                            Bitmap faceBitmap = Bitmap.createBitmap(bmp, bbox.left, bbox.top, bbox.width(), bbox.height());
                            Bitmap resultBitmap = Bitmap.createScaledBitmap(faceBitmap, facialAttributeClassifier.getImageSizeX(), facialAttributeClassifier.getImageSizeY(), false);
                            ClassifierResult res = facialAttributeClassifier.classifyFrame(resultBitmap);
                            ;
                            c.drawText(res.toString(), bbox.left, Math.max(0, bbox.top - 20), p_text);
                            Log.i(TAG, res.toString());
                        }
                    }
                    imageView.setImageBitmap(tempBmp);
                    textView.setText("");
                    textView.append("Find " + (num_clusters) + " clusters\n");
                    textView.append("Cluster " + (clusterized_nums.get(current_num) + 1) + "\n");

                    current_num++;
                }

            }
        });

        Button nextButton = (Button)findViewById(R.id.next_button);
        nextButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (current_num > clusterized_uri.size() - 1) {
                    current_num = 0;
                }
                Uri imgUri = clusterized_uri.get(current_num);
                Bitmap bmp = imagemap.get(imgUri);

                Bitmap resizedBitmap = bmp;
                double minSize = 600.0;
                double scale = Math.min(bmp.getWidth(), bmp.getHeight()) / minSize;
                if (scale > 1.0) {
                    resizedBitmap = Bitmap.createScaledBitmap(bmp, (int) (bmp.getWidth() / scale), (int) (bmp.getHeight() / scale), false);
                    bmp = resizedBitmap;
                }
                Vector<Box> bboxes = mtcnnFaceDetector.detectFaces(resizedBitmap, minFaceSize);//(int)(bmp.getWidth()*MIN_FACE_SIZE));

                Bitmap tempBmp = Bitmap.createBitmap(bmp.getWidth(), bmp.getHeight(), Bitmap.Config.ARGB_8888);
                Canvas c = new Canvas(tempBmp);
                Paint p = new Paint();
                p.setStyle(Paint.Style.STROKE);
                p.setAntiAlias(true);
                p.setFilterBitmap(true);
                p.setDither(true);
                p.setColor(Color.BLUE);
                p.setStrokeWidth(5);

                Paint p_text = new Paint();
                p_text.setColor(Color.WHITE);
                p_text.setStyle(Paint.Style.FILL);
                p_text.setColor(Color.GREEN);
                p_text.setTextSize(24);

                c.drawBitmap(bmp, 0, 0, null);

                for (Box box : bboxes) {

                    p.setColor(Color.RED);
                    android.graphics.Rect bbox = new android.graphics.Rect(bmp.getWidth() * box.left() / resizedBitmap.getWidth(),
                            bmp.getHeight() * box.top() / resizedBitmap.getHeight(),
                            bmp.getWidth() * box.right() / resizedBitmap.getWidth(),
                            bmp.getHeight() * box.bottom() / resizedBitmap.getHeight()
                    );

                    c.drawRect(bbox, p);

                    if (facialAttributeClassifier != null) {
                        Bitmap faceBitmap = Bitmap.createBitmap(bmp, bbox.left, bbox.top, bbox.width(), bbox.height());
                        Bitmap resultBitmap = Bitmap.createScaledBitmap(faceBitmap, facialAttributeClassifier.getImageSizeX(), facialAttributeClassifier.getImageSizeY(), false);
                        ClassifierResult res = facialAttributeClassifier.classifyFrame(resultBitmap);
                        ;
                        c.drawText(res.toString(), bbox.left, Math.max(0, bbox.top - 20), p_text);
                        Log.i(TAG, res.toString());
                    }
                }
                imageView.setImageBitmap(tempBmp);
                textView.setText("");
                textView.append("Find " + (num_clusters) + " clusters\n");
                textView.append("Cluster " + (clusterized_nums.get(current_num) + 1) + "\n");
                current_num++;

            }
        });

        Button prevButton = (Button)findViewById(R.id.prev_button);
        prevButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (current_num < 0)
                {current_num = clusterized_uri.size() - 1;}
                Uri imgUri = clusterized_uri.get(current_num);
                Bitmap bmp = imagemap.get(imgUri);

                Bitmap resizedBitmap = bmp;
                double minSize = 600.0;
                double scale = Math.min(bmp.getWidth(), bmp.getHeight()) / minSize;
                if (scale > 1.0) {
                    resizedBitmap = Bitmap.createScaledBitmap(bmp, (int) (bmp.getWidth() / scale), (int) (bmp.getHeight() / scale), false);
                    bmp = resizedBitmap;
                }
                Vector<Box> bboxes = mtcnnFaceDetector.detectFaces(resizedBitmap, minFaceSize);//(int)(bmp.getWidth()*MIN_FACE_SIZE));

                Bitmap tempBmp = Bitmap.createBitmap(bmp.getWidth(), bmp.getHeight(), Bitmap.Config.ARGB_8888);
                Canvas c = new Canvas(tempBmp);
                Paint p = new Paint();
                p.setStyle(Paint.Style.STROKE);
                p.setAntiAlias(true);
                p.setFilterBitmap(true);
                p.setDither(true);
                p.setColor(Color.BLUE);
                p.setStrokeWidth(5);

                Paint p_text = new Paint();
                p_text.setColor(Color.WHITE);
                p_text.setStyle(Paint.Style.FILL);
                p_text.setColor(Color.GREEN);
                p_text.setTextSize(24);

                c.drawBitmap(bmp, 0, 0, null);

                for (Box box : bboxes) {

                    p.setColor(Color.RED);
                    android.graphics.Rect bbox = new android.graphics.Rect(bmp.getWidth() * box.left() / resizedBitmap.getWidth(),
                            bmp.getHeight() * box.top() / resizedBitmap.getHeight(),
                            bmp.getWidth() * box.right() / resizedBitmap.getWidth(),
                            bmp.getHeight() * box.bottom() / resizedBitmap.getHeight()
                    );

                    c.drawRect(bbox, p);

                    if (facialAttributeClassifier != null) {
                        Bitmap faceBitmap = Bitmap.createBitmap(bmp, bbox.left, bbox.top, bbox.width(), bbox.height());
                        Bitmap resultBitmap = Bitmap.createScaledBitmap(faceBitmap, facialAttributeClassifier.getImageSizeX(), facialAttributeClassifier.getImageSizeY(), false);
                        ClassifierResult res = facialAttributeClassifier.classifyFrame(resultBitmap);;
                        c.drawText(res.toString(), bbox.left, Math.max(0, bbox.top - 20), p_text);
                        Log.i(TAG, res.toString());
                    }
                    imageView.setImageBitmap(tempBmp);
                    textView.setText("");
                    textView.append("Find " + (num_clusters) + " clusters\n");
                    textView.append("Cluster " + (clusterized_nums.get(current_num) + 1) + "\n");

                    current_num--;
            }}
        });

    }

    private void init() throws IOException {
        try {
            mtcnnFaceDetector = MTCNNModel.Companion.create(getAssets());
        } catch (final Exception e) {
            Log.e(TAG, "Exception initializing MTCNNModel!" + e);
        }
        try {
            facialAttributeClassifier=new AgeGenderEthnicityTfLiteClassifier(getApplicationContext());
        } catch (final Exception e) {
            Log.e(TAG, "Exception initializing AgeGenderEthnicityTfLiteClassifier!", e);
        }
        try{
            String t = assetFilePath(this, "resnet.pt");
        module = Module.load(t);
        }
        catch (final Exception e) {
             Log.e(TAG, "Exception initializing AgeGenderEthnicityTfLiteClassifier!", e);
        }
    }

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }
        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
    @RequiresApi(api = Build.VERSION_CODES.P)
    public Bitmap loadFromUri(Uri photoUri) {
        Bitmap image = null;
        try {
            // check version of Android on device
            if(Build.VERSION.SDK_INT > 27){
                // on newer versions of Android, use the new decodeBitmap method
                ImageDecoder.Source source = ImageDecoder.createSource(this.getContentResolver(), photoUri);
                image = ImageDecoder.decodeBitmap(source);
            } else {
                // support older versions of Android by using getBitmap
                image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), photoUri);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return image;
    }
    @RequiresApi(api = Build.VERSION_CODES.P)
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (data.getClipData() != null) {
            ClipData mClipData = data.getClipData();
            mArrayUri = new ArrayList<Uri>();
            mBitmapsSelected = new ArrayList<Bitmap>();
            result_feats = new ArrayList<List<FaceFeatures>>();
            HashMap<FaceFeatures, String> featsMap = new HashMap<>();
            boxmap = new HashMap<>();
            imagemap = new HashMap<>();
            for (int i = 0; i < mClipData.getItemCount(); i++) {
                ClipData.Item item = mClipData.getItemAt(i);
                Uri uri = item.getUri();
                mArrayUri.add(uri);
                Bitmap bitmap = loadFromUri(uri).copy(Bitmap.Config.RGB_565, true);
                mBitmapsSelected.add(bitmap);
                Bitmap bmp = bitmap;
                Bitmap resizedBitmap = bmp;
                double minSize = 600.0;
                double scale = Math.min(bmp.getWidth(), bmp.getHeight()) / minSize;
                if (scale > 1.0) {
                    resizedBitmap = Bitmap.createScaledBitmap(bmp, (int) (bmp.getWidth() / scale), (int) (bmp.getHeight() / scale), false);
                    bmp = resizedBitmap;
                }
                Vector<Box> bboxes = mtcnnFaceDetector.detectFaces(resizedBitmap, minFaceSize);//(int)(bmp.getWidth()*MIN_FACE_SIZE));
                boxmap.put(uri, bboxes);
                imagemap.put(uri, bitmap);
                }
        }
    }

    private List<FaceFeatures> getFacesFeatures(Bitmap bmp){
        Bitmap resizedBitmap=bmp;
        double minSize=600.0;
        double scale=Math.min(bmp.getWidth(),bmp.getHeight())/minSize;
        if(scale>1.0) {
            resizedBitmap = Bitmap.createScaledBitmap(bmp, (int)(bmp.getWidth()/scale), (int)(bmp.getHeight()/scale), false);
            bmp=resizedBitmap;
        }
        List<FaceFeatures> facesInfo=new ArrayList<>();
        try
        {
        Vector<Box> bboxes = mtcnnFaceDetector.detectFaces(resizedBitmap, minFaceSize);//(int)(bmp.getWidth()*MIN_FACE_SIZE));

        for (Box box : bboxes) {
            android.graphics.Rect bbox = new android.graphics.Rect(bmp.getWidth()*box.left() / resizedBitmap.getWidth(),
                    bmp.getHeight()* box.top() / resizedBitmap.getHeight(),
                    bmp.getWidth()* box.right() / resizedBitmap.getWidth(),
                    bmp.getHeight() * box.bottom() / resizedBitmap.getHeight()
            );
            Bitmap faceBitmap = Bitmap.createBitmap(bmp, bbox.left, bbox.top, bbox.width(), bbox.height());
            Bitmap resultBitmap = Bitmap.createScaledBitmap(faceBitmap, 224, 224, false);
            FaceData res=(FaceData)facialAttributeClassifier.classifyFrame(resultBitmap);
            facesInfo.add(new FaceFeatures(res.features,0.5f*(box.left()+box.right()) / resizedBitmap.getWidth(),0.5f*(box.top()+box.bottom()) / resizedBitmap.getHeight()));
        }
        }
        catch (Exception e)
        {
            return facesInfo;
        }

        return facesInfo;
    }
    private class FaceFeatures{
        public FaceFeatures(float[] feat, float x, float y){
            features=feat;
            centerX=x;
            centerY=y;
        }
        public float[] features;
        public float centerX,centerY;
    }
    private boolean allPermissionsGranted() {
        for (String permission : getRequiredPermissions()) {
            int status= ContextCompat.checkSelfPermission(this,permission);
            if (ContextCompat.checkSelfPermission(this,permission)
                    != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }
    public class DistanceMetricFaceFeatures implements DistanceMetric<FaceFeatures> {

        @Override
        public double calculateDistance(FaceFeatures val1, FaceFeatures val2) {
            double dist = 0;
            for (int i = 0; i < val1.features.length; ++i) {
                dist += (val1.features[i] - val2.features[i]) * (val1.features[i] - val2.features[i]);
            }
            return Math.sqrt(dist);
        }
    }

    public class DistanceMetricFloat implements DistanceMetric<float[]> {

        @Override
        public double calculateDistance(float[] val1, float[] val2) throws DBSCANClusteringException {
            double dist = 0;
            for (int i = 0; i < val1.length; ++i) {
                dist += (val1[i] - val2[i]) * (val1[i] - val2[i]);
            }
            return Math.sqrt(dist);
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
    }

    private String[] getRequiredPermissions() {
        try {
            PackageInfo info =
                    getPackageManager()
                            .getPackageInfo(getPackageName(), PackageManager.GET_PERMISSIONS);
            String[] ps = info.requestedPermissions;
            if (ps != null && ps.length > 0) {
                return ps;
            } else {
                return new String[0];
            }
        } catch (Exception e) {
            return new String[0];
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        switch (requestCode) {
            case REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS:
                Map<String, Integer> perms = new HashMap<String, Integer>();
                boolean allGranted = true;
                for (int i = 0; i < permissions.length; i++) {
                    perms.put(permissions[i], grantResults[i]);
                    if (grantResults[i] != PackageManager.PERMISSION_GRANTED)
                        allGranted = false;
                }
                // Check for ACCESS_FINE_LOCATION
                if (allGranted) {
                    // All Permissions Granted
                    try {
                        init();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                } else {
                    // Permission Denied
                    Toast.makeText(MainActivity.this, "Some Permission is Denied", Toast.LENGTH_SHORT)
                            .show();
                    finish();
                }
                break;
            default:
                super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }
}