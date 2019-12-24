import javafx.application.Application;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.image.PixelWriter;
import javafx.stage.Stage;
import javafx.scene.paint.Color;

 
public class GraphicViewer extends Application {
	private static final int WIDTH = 400;
	private static final int HEIGHT = 400;
	public static void main(String[] args) {
		Application.launch(args);
	}
	@Override
	public void start(Stage stage) { // invoked when the application starts
		stage.setTitle("Mandelbrot");
		Canvas canvas = new Canvas(WIDTH, HEIGHT);
		GraphicsContext gc = canvas.getGraphicsContext2D();
		paintScene(gc.getPixelWriter(), WIDTH, HEIGHT);
		stage.setScene(new Scene(new Group(canvas)));
		stage.show();
	}
	
	
	private void paintScene(PixelWriter wr, int width, int height) {
		// draw a picture on g
		final Complex CENTER = new Complex(-0.5,0.0);
		final int MAX_ITER = 100;
		final int R = 2;
		double MODEL_WIDTH = 3.0;
		
		Complex C = new Complex(0.0,0.0);
		Complex Z = new Complex(0.0,0.0);
		double dx = MODEL_WIDTH / width;
		for (int y=0;y<height;y++) {
			for(int x=0; x<width;x++) {
				C.setReal((x-width/2.0)*dx);
				C.setImg((height/2.0-y)*dx);
				C.AddUp(CENTER);
				Z.setReal(0.0);Z.setImg(0.0); int iter = 0;
				do {
					Z.square();Z.AddUp(C);iter++;
				}while((iter<MAX_ITER) && (Z.abs()<R));
				double color= (double)iter/(double)MAX_ITER;
				Color col = Color.hsb(color, 1.0, 1.0);
				wr.setColor(x, y, col);		
			}
			
		}
	}
}










