package jdraw.figures;

import java.awt.Point;

import jdraw.framework.DrawContext;

public class CircleTool extends AbstractTool{


	public CircleTool(DrawContext context, String Name, String Icon) {
		super(context);
		this.name = Name;
		this.icon = Icon;
	}


	@Override
	protected AbstractFigure createFigure(Point p) {
		return new Circle(p);
	}

}
