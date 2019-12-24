package graphics;
import javafx.application.Application;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.image.PixelWriter;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

public class GraphicViewerFX extends Application {
	private static final int WIDTH = 400;
	private static final int HEIGHT = 400;

	public static void main(String[] args) {
		Application.launch(args);
	}

	@Override
	public void start(Stage stage) {
		stage.setTitle("Graphics Viewer");
		Canvas canvas = new Canvas(WIDTH, HEIGHT);
		GraphicsContext gc = canvas.getGraphicsContext2D();
		paintScene(gc.getPixelWriter(), WIDTH, HEIGHT);
		stage.setScene(new Scene(new Group(canvas)));
		stage.show();
	}

	private void paintScene(PixelWriter wr, int width, int height) {
		for (int x = 0; x < width; x++) {
			int y = (int) (height / 2  * (1 - Math.sin(6 * Math.PI / width * x)));
			Color c = Color.hsb(360*x / ((float) width), 1.0f, 1.0f);
			for(int p = height/2; p <= y; p++) wr.setColor(x, p, c);
			for(int p = y; p <= height/2; p++) wr.setColor(x, p, c);
		}
	}

}
