package jdraw.figures;

import java.awt.Color;
import java.awt.Graphics;

@SuppressWarnings("serial")
public class Oval extends AbstractRectangularFigure{
	/**
	 * Use the java.awt.Rectangle in order to save/reuse code.
	 */

	
	/**
	 * Create a new rectangle of the given dimension.
	 * @param x the x-coordinate of the upper left corner of the rectangle
	 * @param y the y-coordinate of the upper left corner of the rectangle
	 * @param w the rectangle's width
	 * @param h the rectangle's height
	 */
	public Oval(int x, int y, int w, int h) {
		super(x, y, w, h);
	}

	/**
	 * Draw the rectangle to the given graphics context.
	 * @param g the graphics context to use for drawing.
	 */
	@Override
	public void draw(Graphics g) {
		g.setColor(Color.WHITE);
		g.fillOval(rectangle.x, rectangle.y, rectangle.width, rectangle.height);
		g.setColor(Color.BLACK);
		g.drawOval(rectangle.x, rectangle.y, rectangle.width, rectangle.height);
	}
	


	
}
