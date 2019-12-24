package jdraw.figures;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Point;

@SuppressWarnings("serial")
public class Circle extends AbstractRectangularFigure{

	public Circle(int x, int y, int w, int h) {
		super(x, y, h, h);
	}

	public Circle(Point p) {
		super(p.x, p.y, 0,0);
	}

	@Override
	public void draw(Graphics g) {
		g.setColor(Color.WHITE);
		g.fillOval(rectangle.x, rectangle.y, rectangle.width, rectangle.width);
		g.setColor(Color.BLACK);
		g.drawOval(rectangle.x, rectangle.y, rectangle.width, rectangle.width);		
	}

}
