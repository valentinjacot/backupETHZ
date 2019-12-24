import javafx.application.Application;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.image.PixelWriter;
import javafx.stage.Stage;
import javafx.scene.paint.Color;

 
public class GraphicViewerFX extends Application {
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
		int max = 40;
		double dx = 3.0/(double)width;
		Complex z = new Complex(0.0,0.0);
		Complex Z0 = new Complex(0.0,0.0);
		for (int m=0;m<width;m++) {
			for (int n=0;n<height;n++) {
				double x0 = (m-width/2.0)*dx - 0.5;
				double y0 = (height/2.0-n)*dx;
				//Complex Z0= new Complex(x0,y0);
				Z0.setReal(x0);
				Z0.setImg(y0);
				//double iter= (double)mand(Z0,max)/(double)max;
				int i=0;
				do {
					z = Complex.sum(z.square(), Z0);
					i++;
				}while((z.abs()<2.0) & (i<max));
				double iter = 255 - (double)i/(double)max;
				iter /=255;
				//System.out.println(iter);
				Color color =Color.hsb(iter,1.0f,1.0f);
				wr.setColor(m, n, color);		
			}
		}		
	}
}