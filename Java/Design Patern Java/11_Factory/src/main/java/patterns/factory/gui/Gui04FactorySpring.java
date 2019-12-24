package patterns.factory.gui;

import org.springframework.context.support.ClassPathXmlApplicationContext;

import patterns.factory.gui.Components.Frame;

public class Gui04FactorySpring {
	private static ClassPathXmlApplicationContext ctx;

	static {
		ctx = new ClassPathXmlApplicationContext("gui-context.xml");
	}

	public static void main(String[] args) {
		CalculatorFactory calcFactory = ctx.getBean("calculatorFactoryBean", CalculatorFactory.class);
		Frame f = calcFactory.newCalculatorFrame();
		f.setVisible(true);
	}
}
