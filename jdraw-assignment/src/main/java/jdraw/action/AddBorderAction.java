package jdraw.action;

import java.awt.event.ActionEvent;
import java.util.List;

import javax.swing.Action;
import javax.swing.JMenu;
import javax.swing.event.MenuEvent;
import javax.swing.event.MenuListener;

import jdraw.decorators.BorderedDecorator;
import jdraw.framework.DrawContext;
import jdraw.framework.DrawModel;
import jdraw.framework.Figure;

@SuppressWarnings("serial")
public class AddBorderAction extends AbstractAction implements MenuListener{

	public AddBorderAction(DrawContext AC, JMenu JM) {
		super(AC);
//		putValue(Action.SMALL_ICON, new ImageIcon(getClass().getResource(IMAGES + "group.png")));
		putValue(Action.SHORT_DESCRIPTION, "Decorates the firgure with a border");
		putValue(Action.NAME, "Border");
		JM.addMenuListener(this);
    }
	
	@Override
	public void menuSelected(MenuEvent ignore) {
		setEnabled(actionContext.getView().getSelection().size() > 0);
	}
	
	@Override
	public void actionPerformed(ActionEvent e) {
		List<Figure> selection = actionContext.getView().getSelection();
		if(selection !=null && selection.size()>0) {
			DrawModel dm = actionContext.getView().getModel();
			for(Figure f : selection ) {
				actionContext.getView().removeFromSelection(f);
				dm.removeFigure(f);
				BorderedDecorator BD = new BorderedDecorator(f);
				dm.addFigure(BD);
				actionContext.getView().addToSelection(BD);
			}
		}
	}

}
