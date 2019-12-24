package jdraw.toolfactory;

import jdraw.figures.GreenCircleTool;
import jdraw.framework.DrawContext;
import jdraw.framework.DrawTool;

public class GreenCircleToolFactory extends AbstractDrawToolFactory{

	@Override
	public DrawTool createTool(DrawContext context) {
		return new GreenCircleTool(context, getName(), getIconName());
	}

}
