package graphics;
import javafx.application.Application;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.image.PixelWriter;
import javafx.scene.paint.Color;
import javafx.stage.Stage;
import java.lang.Math;


public class Mandelbrot extends Application {
	private static final int WIDTH = 800;
	private static final int HEIGHT = 800;

	public static void main(String[] args) {
		Application.launch(args);
	}

	@Override
	public void start(Stage stage) {
		stage.setTitle("Mandelbrot");
		Canvas canvas = new Canvas(WIDTH, HEIGHT);
		GraphicsContext gc = canvas.getGraphicsContext2D();
		paintScene(gc.getPixelWriter(), WIDTH, HEIGHT);
		stage.setScene(new Scene(new Group(canvas)));
		stage.show();
	}

	private void paintScene(PixelWriter wr, int width, int height) {
		Complex z1 = new Complex(Complex.ONE);
		//Complex z = new Complex(Complex.ZERO);
		double dx = 3./(double)width;
		int max = 100;
		for (int m=0;m<width;m++) {
			for (int n=0;n<height;n++) {
				double x0 = (m-width/2.0)*dx - 0.5;
				double y0= (height/2.0 -n)*dx;
				double x =0;
				double y=0;
				Complex z = new Complex(x,y);
				double iter =0;
				double itersmooth = Math.exp(-z.abs());
				do {
					double xtemp = z.square().getReal() + x0;
					y = z.square().getImg() + y0;
					x= xtemp;
					//z = Complex.sum(z.square(),z1);
					z = new Complex(x,y);
					iter+=1;
					itersmooth += Math.exp(-z.abs());
				}while(iter<max && z.abs()<2);
//				itersmooth = (255- iter)/iter;        			// uncomment for the "non-smoothed" image, 
				itersmooth = (100 - itersmooth)/itersmooth; 	//and comment this one
				//double itersmooth = iter + 	1  - Math.log(Math.log(z.abs()))/Math.log(2);
				Color color = Color.hsb(itersmooth,1.0f, 1.0f);
				wr.setColor(m, n, color);
				
			}
		}
		
	}

}