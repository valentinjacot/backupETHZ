package jdraw.figures;

import java.awt.Point;

import jdraw.framework.DrawContext;

public class LineTool extends AbstractTool {
	

		public LineTool(DrawContext context, String Name, String Icon) {
			super(context);
			this.name = Name;
			this.icon = Icon;
		}
		@Override
		protected AbstractFigure createFigure(Point p) {
			return new Line(p.x,p.y, p.x, p.y);
		}
	}