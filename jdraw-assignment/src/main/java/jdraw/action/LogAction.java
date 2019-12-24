package jdraw.action;

import java.awt.event.ActionEvent;
import java.util.List;

import javax.swing.Action;
import javax.swing.JMenu;
import javax.swing.event.MenuListener;

import jdraw.decorators.LogDecorator;
import jdraw.framework.DrawContext;
import jdraw.framework.DrawModel;
import jdraw.framework.Figure;

@SuppressWarnings("serial")
public class LogAction extends AbstractAction implements MenuListener{

	public LogAction(DrawContext AC, JMenu JM) {
		super(AC);
//		putValue(Action.SMALL_ICON, new ImageIcon(getClass().getResource(IMAGES + "group.png")));
		putValue(Action.SHORT_DESCRIPTION, "Adds a Log to the figure ");
		putValue(Action.NAME, "Log");
		JM.addMenuListener(this);
    }

	@Override
	public void actionPerformed(ActionEvent e) {
		List<Figure> selection = actionContext.getView().getSelection();
		if(selection !=null && selection.size()>0) {
			DrawModel dm = actionContext.getView().getModel();
			for(Figure f : selection ) {
				actionContext.getView().removeFromSelection(f);
				dm.removeFigure(f);
				LogDecorator AD = new LogDecorator(f);
				dm.addFigure(AD);
				actionContext.getView().addToSelection(AD);
			}
		}
	}

}
