package jdraw.toolfactory;

import jdraw.figures.LineTool;
import jdraw.framework.DrawContext;
import jdraw.framework.DrawTool;

public class LineToolFactory extends AbstractDrawToolFactory{

	@Override
	public DrawTool createTool(DrawContext context) {
		return new LineTool(context, getName(), getIconName());
	}

}
