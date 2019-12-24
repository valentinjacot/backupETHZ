package patterns.factory.gui;

import patterns.factory.gui.Components.ActionListener;
import patterns.factory.gui.Components.Button;
import patterns.factory.gui.Components.Component;
import patterns.factory.gui.Components.Field;
import patterns.factory.gui.Components.Frame;


public class CalculatorFactoryImpl implements CalculatorFactory {

	private ComponentFactory componentFactory;

	public void setComponentFactory(ComponentFactory fact){
		this.componentFactory = fact;
	}

	private String title = "Calculator";

	public void setTitle(String title) {
		this.title = title;
	}

	@Override
	public Frame newCalculatorFrame() {

		Frame f = componentFactory.newFrame(title);

		final Field x   = componentFactory.newField(10, true);
		final Field y   = componentFactory.newField(10, true);
		final Field sum = componentFactory.newField(10, false);

		Button b = componentFactory.newButton("Compute",
			new ActionListener(){
				@Override
				public void actionPerformed(Component source){
					int ix = Integer.parseInt(x.getText());
					int iy = Integer.parseInt(y.getText());
					sum.setText("" + (ix + iy));
				}
			}
		);

		f.setGrid(4, 2);

		f.add(componentFactory.newLabel("x"));   f.add(x);
		f.add(componentFactory.newLabel("y"));   f.add(y);
		f.add(componentFactory.newLabel("sum")); f.add(sum);
		f.add(b);

		return f;
	}
}

