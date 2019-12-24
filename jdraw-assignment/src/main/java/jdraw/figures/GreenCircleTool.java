package jdraw.figures;

import java.awt.Point;

import jdraw.framework.DrawContext;

public class GreenCircleTool extends CircleTool{

	public GreenCircleTool(DrawContext context,String Name, String Icon) {
		super(context, Name, Icon);
		this.name = Name;
		this.icon = Icon;
	}
	@Override
	protected AbstractFigure createFigure(Point p) {
		return new CircleGreen(p.x,p.y,p.x,p.x);
	}

}
