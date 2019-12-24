package graphics;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.WindowConstants;

@SuppressWarnings("serial")
public class GraphicViewerSwing extends JPanel {

	private int width = 400;
	private int height = 400;

	public GraphicViewerSwing() {
		setBackground(Color.white);
		setPreferredSize(new Dimension(width, height));
	}

	public void paintComponent(Graphics g) {
		super.paintComponent(g);
		
		g.setColor(Color.red);
		int width = getWidth();
		int height = getHeight();
		int oldy = height / 2;
		for (int x = 1; x < width; x++) {
			int y = (int) (height * 0.5 * (1 - Math
					.sin(6 * Math.PI / width * x)));
			g.setColor(Color.getHSBColor(x / ((float) width), 1.0f, 1.0f));
			g.drawLine(x - 1, oldy, x, y);
			oldy = y;
		}
	}

	public static void main(String[] args) {
		JFrame f = new JFrame("Graphic Viewer");
		f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		f.setResizable(false);
		f.add(new GraphicViewerSwing());
		f.pack();
		f.setVisible(true);
	}

}
