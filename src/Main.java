import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class Main {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME); // Load OpenCV library
    }

    public static void main(String[] args) {
        // Initialize face detector
        CascadeClassifier faceDetector = new CascadeClassifier("C:\\Users\\PRIYANSHU\\IdeaProjects\\face - Copy\\src\\haarcascade_frontalface_default.xml");

        // Load known faces
        Map<String, Mat> knownFaces = loadKnownFaces("C:\\Users\\PRIYANSHU\\IdeaProjects\\face - Copy\\images");

        // Open the webcam
        VideoCapture camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.out.println("Error: Camera not detected!");
            return;
        }

        Mat frame = new Mat();
        boolean running = true;
        while (running) {
            if (camera.read(frame)) {
                Mat grayFrame = new Mat();
                Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);

                // Detect faces
                Rect[] faces = detectFaces(faceDetector, grayFrame);
                if (faces.length > 0) {
                    System.out.println("Faces detected: " + faces.length);  // Debugging
                }

                for (Rect face : faces) {
                    Mat faceROI = grayFrame.submat(face);

                    // Preprocess the detected face
                    Mat processedFace = preprocessFace(faceROI);
                    System.out.println("Processed face size: " + processedFace.size());  // Debugging

                    // Compare the processed face with known faces manually using template matching
                    String name = recognizeFace(processedFace, knownFaces);

                    // Draw a rectangle and display the name
                    Imgproc.rectangle(frame, face, new Scalar(0, 255, 0));
                    Imgproc.putText(frame, name, new Point(face.x, face.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 255, 0));

                    System.out.println("Recognized Name: " + name);  // Debugging
                }

                // Display the video feed
                HighGui.imshow("Face Recognition", frame);

                // Break the loop if the user presses the 'ESC' key
                int key = HighGui.waitKey(1);
                if (key == 27) {  // ESC key code
                    running = false;
                }
            }
        }

        // Release resources after the loop ends
        camera.release();
        HighGui.destroyAllWindows();
    }

    private static Map<String, Mat> loadKnownFaces(String folderPath) {
        Map<String, Mat> faceMap = new HashMap<>();
        File folder = new File(folderPath);
        File[] files = folder.listFiles((dir, name) -> name.toLowerCase().endsWith(".jpg") || name.toLowerCase().endsWith(".png") || name.toLowerCase().endsWith(".jpeg"));

        // Check if files are present
        if (files != null) {
            for (File file : files) {
                // Extract the name without extension
                String name = file.getName().substring(0, file.getName().lastIndexOf('.'));
                // Read the image in grayscale
                Mat face = Imgcodecs.imread(file.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);

                // If the image is loaded correctly, add it to the map
                if (!face.empty()) {
                    faceMap.put(name, preprocessFace(face));
                    System.out.println("Loaded known face: " + name);  // Debugging
                }
            }
        }
        return faceMap;
    }

    private static Rect[] detectFaces(CascadeClassifier detector, Mat frame) {
        MatOfRect faceDetections = new MatOfRect();
        detector.detectMultiScale(frame, faceDetections);
        return faceDetections.toArray();
    }

    private static Mat preprocessFace(Mat face) {
        Mat resizedFace = new Mat();
        Imgproc.resize(face, resizedFace, new Size(120, 120)); // Resize to a consistent size
        Imgproc.equalizeHist(resizedFace, resizedFace);  // Apply histogram equalization for better contrast
        return resizedFace;
    }

    private static String recognizeFace(Mat processedFace, Map<String, Mat> knownFaces) {
        double maxSimilarity = 0.0;
        String recognizedName = "Unknown";
        double threshold = 0.4;  // Lower threshold for better matching tolerance

        // Iterate over the known faces
        for (Map.Entry<String, Mat> entry : knownFaces.entrySet()) {
            // Apply template matching at different scales
            for (double scale = 0.8; scale <= 1.2; scale += 0.1) {
                Mat scaledFace = new Mat();
                Size newSize = new Size(processedFace.width() * scale, processedFace.height() * scale);
                Imgproc.resize(processedFace, scaledFace, newSize);

                // Perform template matching on the scaled face
                double similarity = compareFaces(scaledFace, entry.getValue());
                System.out.println("Similarity score for " + entry.getKey() + " at scale " + scale + ": " + similarity);

                // If the similarity is above the threshold, update recognized name
                if (similarity > maxSimilarity && similarity > threshold) {
                    maxSimilarity = similarity;
                    recognizedName = entry.getKey();
                }
            }
        }

        return recognizedName;
    }

    private static double compareFaces(Mat face1, Mat face2) {
        // Use Normalized Cross-Correlation (NCC) for template matching
        Mat result = new Mat();
        Imgproc.matchTemplate(face1, face2, result, Imgproc.TM_CCOEFF_NORMED);

        // Get the similarity score (maximum value)
        Core.MinMaxLocResult minMaxLocResult = Core.minMaxLoc(result);
        return minMaxLocResult.maxVal;  // Return the similarity score
    }
}
