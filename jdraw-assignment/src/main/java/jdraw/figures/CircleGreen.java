package jdraw.figures;

import java.awt.Color;
import java.awt.Graphics;

@SuppressWarnings("serial")
public class CircleGreen extends Circle{

	public CircleGreen(int x, int y, int w, int h) {
		super(x, y, w, h);
	}
	@Override
	public void draw(Graphics g) {
		g.setColor(Color.GREEN);
		g.fillOval(rectangle.x, rectangle.y, rectangle.width, rectangle.width);
		g.setColor(Color.GREEN);
		g.drawOval(rectangle.x, rectangle.y, rectangle.width, rectangle.width);		
	}
}
