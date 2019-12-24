package jdraw.toolfactory;

import jdraw.figures.RectTool;
import jdraw.framework.DrawContext;
import jdraw.framework.DrawTool;

public class RectToolFactory extends AbstractDrawToolFactory{

	@Override
	public DrawTool createTool(DrawContext context) {
		return new RectTool(context, getName(), getIconName());
	}

}
