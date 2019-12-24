/*
 * Copyright (c) 2018 Fachhochschule Nordwestschweiz (FHNW)
 * All Rights Reserved. 
 */

package jdraw.figures;

import java.awt.Point;

import jdraw.framework.DrawContext;

/**
 * This tool defines a mode for drawing rectangles.
 *
 * @see jdraw.framework.Figure
 *
 * @author  Christoph Denzler
 */
public class RectTool extends AbstractTool {
  
	public RectTool(DrawContext context, String Name, String Icon) {
		super(context);
		this.name = Name;
		this.icon = Icon;
	}

	@Override
	protected AbstractFigure createFigure(Point p) {
		// TODO Auto-generated method stub
		return new Rect(p.x, p.y, 0,0);
	}

	
}
