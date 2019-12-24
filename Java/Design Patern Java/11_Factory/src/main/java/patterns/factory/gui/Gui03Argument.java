package patterns.factory.gui;

import patterns.factory.gui.Components.Component;
import patterns.factory.gui.Components.Field;
import patterns.factory.gui.Components.Frame;

// generic solution of calculator where factory
// can be specified with a runtime argument
/////////////////////////////////////////////////

public class Gui03Argument {

	public static void main(String[] args) throws Exception {
		Class.forName("patterns.factory.gui." + args[0]);

		ComponentFactory componentFactory = CurrentFactory.getFactory();
		Frame f = componentFactory.newFrame("Calculator");
		final Field x = componentFactory.newField(10, true);
		final Field y = componentFactory.newField(10, true);
		final Field sum = componentFactory.newField(10, false);
		Component b = componentFactory.newButton("Compute", source -> {
			int ix = Integer.parseInt(x.getText());
			int iy = Integer.parseInt(y.getText());
			sum.setText("" + (ix + iy));
		});
		f.setGrid(4, 2);
		f.add(componentFactory.newLabel("x"));
		f.add(x);
		f.add(componentFactory.newLabel("y"));
		f.add(y);
		f.add(componentFactory.newLabel("sum"));
		f.add(sum);
		f.add(b);
		f.setVisible(true);
	}

}
