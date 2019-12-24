package jdraw.figures;

import java.awt.Point;

import jdraw.framework.DrawContext;

public class OvalTool extends AbstractTool{

		public OvalTool(DrawContext context, String Name, String Icon) {
			super(context);
			this.name = Name;
			this.icon = Icon;
		}

		@Override
		protected AbstractFigure createFigure(Point p) {
			// TODO Auto-generated method stub
			return new Oval(p.x,p.y,0,0);
		}
		
	}
