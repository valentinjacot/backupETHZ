
package patterns.strategy.layout;

import java.awt.*;

public class ListLayout implements LayoutManager {
		
	@Override
	public void addLayoutComponent(String name, Component comp) { }

	@Override
	public void removeLayoutComponent(Component comp) { }
	
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
		Insets insets = parent.getInsets();
		int x = insets.left;
		int y = insets.top;
		int w = parent.getSize().width -insets.left-insets.right;
		
		int numberOfComponents=parent.getComponentCount();
		for (int i=0; i<numberOfComponents; i++){
			Component c = parent.getComponent(i);
			if (c!=null && c.isVisible()) 
				c.setBounds(x, y, w, c.getPreferredSize().height);
			y += c.getPreferredSize().height;
		}			
	}
}