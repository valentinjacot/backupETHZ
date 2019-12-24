package jdraw.figures;

import java.awt.Cursor;
import java.awt.Point;
import java.awt.event.MouseEvent;

import javax.swing.Icon;
import javax.swing.ImageIcon;

import jdraw.commands.AddFigureCommand;
import jdraw.framework.DrawContext;
import jdraw.framework.DrawTool;
import jdraw.framework.DrawView;

public abstract class AbstractTool  implements DrawTool {
	  
		/** 
		 * the image resource path. 
		 */
		protected static final String IMAGES = "/images/";
		protected String name;
		protected String icon;
		private AbstractFigure newRect ;
		 
		/**
		 * Create a new rectangle tool for the given context.
		 * @param context a context to use this tool in.
		 */
		public AbstractTool(DrawContext context, String Name, String Icon) {
			this.context = context;
			this.view = context.getView();
			this.name = Name;
			this.icon = Icon;
		}		
		
		protected void setName(String name) {
			this.name = name;
		}

		protected void setIcon(String icon) {
			this.icon = icon;
		}

		//shortcut constructor ^^
		public AbstractTool(DrawContext context) {
			this.context = context;
			this.view = context.getView();
		}


		@Override
		public Icon getIcon() {
			return new ImageIcon(getClass().getResource(IMAGES + icon)); 
			////this
		}
		/**
		 * Activates the Rectangle Mode. There will be a
		 * specific menu added to the menu bar that provides settings for
		 * Rectangle attributes
		 */
		@Override
		public void activate() {
			this.context.showStatusText(name + " Mode");  
		}

		@Override
		public String getName() {
			return name;
		}
		/**
		 * The context we use for drawing.
		 */
		protected final DrawContext context;

		/**
		 * The context's view. This variable can be used as a shortcut, i.e.
		 * instead of calling context.getView().
		 */
		protected final DrawView view;


		/**
		 * Temporary variable.
		 * During rectangle creation this variable refers to the point the
		 * mouse was first pressed.
		 */
		protected Point anchor = null;

		/**
		 * Deactivates the current mode by resetting the cursor
		 * and clearing the status bar.
		 * @see jdraw.framework.DrawTool#deactivate()
		 */
		@Override
		public void deactivate() {
			this.context.showStatusText("");
		}


		@Override
		public Cursor getCursor() {
			return Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR);
		}
		
		@Override
		public void mouseDown(int x, int y, MouseEvent e) {
			if (newRect != null) {
				throw new IllegalStateException();
			}
			anchor = new Point(x, y);
			newRect = createFigure(anchor);
			view.getModel().addFigure(newRect);
		}

		@Override
		public void mouseUp(int x, int y, MouseEvent e) {
			this.context.getModel().getDrawCommandHandler().addCommand(
					 new AddFigureCommand(this.context.getModel(), newRect)
			);
			newRect = null;
			anchor = null;
			this.context.showStatusText(name + " Mode");
		}

		
		@Override
		public void mouseDrag(int x, int y, MouseEvent e) {
			newRect.setBounds(anchor, new Point(x, y));
			java.awt.Rectangle r = newRect.getBounds();
			this.context.showStatusText("w: " + r.width + ", h: " + r.height);
		}
		
		protected abstract AbstractFigure createFigure(Point p);

	}
