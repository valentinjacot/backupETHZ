package jdraw.action;

import java.awt.event.ActionEvent;
import java.util.List;

import javax.swing.Action;
import javax.swing.ImageIcon;
import javax.swing.JMenu;
import javax.swing.event.MenuEvent;
import javax.swing.event.MenuListener;

import jdraw.figures.Group;
import jdraw.framework.DrawContext;
import jdraw.framework.DrawModel;
import jdraw.framework.DrawView;
import jdraw.framework.Figure;


@SuppressWarnings("serial")
public class UngroupAction extends AbstractAction implements MenuListener{
	
	public UngroupAction (DrawContext AC, JMenu JM) {
		super(AC);		
		putValue(Action.SMALL_ICON, new ImageIcon(getClass().getResource(IMAGES + "ungroup.png")));
		putValue(Action.SHORT_DESCRIPTION, "Ungroups all selected figures");
		putValue(Action.NAME, "Ungroup");
		JM.addMenuListener(this);
	}
	
	
	@Override
	public void menuSelected(MenuEvent ignore) {
		boolean grouped = false;
		for (Figure f:actionContext.getView().getSelection())
			if (f instanceof Group) {
				grouped = true;
				break;
			}
		setEnabled(grouped);
	}
	
	@Override
	public void actionPerformed(ActionEvent e) {
		DrawView dv = actionContext.getView();
		List<Figure> selection = actionContext.getView().getSelection();
		for (Figure f: selection)
			if(f instanceof Group) {
			Group g = (Group) f;
			DrawModel dm = actionContext.getView().getModel();
			dm.removeFigure(f);
			dv.removeFromSelection(f);
			for(Figure p : g.getFigureParts() ) {
				dm.addFigure(p);
				dv.addToSelection(p);
			}
		}
		
	}

}
