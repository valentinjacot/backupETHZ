package jdraw.action;

import java.awt.event.ActionEvent;
import java.util.List;

import javax.swing.Action;
import javax.swing.JMenu;
import javax.swing.event.MenuEvent;
import javax.swing.event.MenuListener;

import jdraw.decorators.AnimationDecorator;
import jdraw.framework.DrawContext;
import jdraw.framework.DrawModel;
import jdraw.framework.Figure;

@SuppressWarnings("serial")
public class AnimationAction extends AbstractAction implements MenuListener{

	public AnimationAction(DrawContext AC, JMenu JM) {
		super(AC);
//		putValue(Action.SMALL_ICON, new ImageIcon(getClass().getResource(IMAGES + "group.png")));
		putValue(Action.SHORT_DESCRIPTION, "Animates the figure ");
		putValue(Action.NAME, "Aniamtion");
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
				AnimationDecorator AD = new AnimationDecorator(f);
				dm.addFigure(AD);
				actionContext.getView().addToSelection(AD);
			}
		}
	}
}
