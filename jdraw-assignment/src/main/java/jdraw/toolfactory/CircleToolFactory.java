package jdraw.toolfactory;

import jdraw.figures.CircleTool;
import jdraw.framework.DrawContext;
import jdraw.framework.DrawTool;

public class CircleToolFactory extends AbstractDrawToolFactory{

	@Override
	public DrawTool createTool(DrawContext context) {
		return new CircleTool(context, getName(), getIconName());
	}
}
