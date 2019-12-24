package patterns.factory.gui;

import patterns.factory.gui.Components.Button;
import patterns.factory.gui.Components.Field;
import patterns.factory.gui.Components.Frame;

// generic solution of calculator
// (i.e. factory can be changed from FactorySwing
// to FactoryAWT)
/////////////////////////////////////////////////

public class Gui02FactoryFixed {

	public static void main(String[] args) {
		ComponentFactory componentFactory = new FactorySWT();
		Frame f = componentFactory.newFrame("Calculator");
		final Field x = componentFactory.newField(10, true);
		final Field y = componentFactory.newField(10, true);
		final Field sum = componentFactory.newField(10, false);
		Button b = componentFactory.newButton("Compute", source -> {
			int ix = Integer.parseInt(x.getText());
			int iy = Integer.parseInt(y.getText());
			sum.setText("" + (ix + iy));
		});
		f.setGrid(4, 2);
		f.add(componentFactory.newLabel("x"));
		f.add(x);
		f.add(componentFactory.newLabel("y"));
		f.add(y);
		f.add(componentFactory.newLabel("Summe"));
		f.add(sum);
		f.add(b);
		f.setVisible(true);
	}

}
