package patterns.strategy.layout;

import java.awt.Component;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.LayoutManager;

public class NullLayout implements LayoutManager {

	@Override
	public void addLayoutComponent(String name, Component comp) {
	}

	@Override
	public void removeLayoutComponent(Component comp) {
	}

	@Override
	public Dimension minimumLayoutSize(Container parent) {
		return parent.getSize();
	}

	@Override
	public Dimension preferredLayoutSize(Container parent) {
		return parent.getSize();
	}

	@Override
	public void layoutContainer(Container parent) {
	}
}
