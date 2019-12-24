package jdraw.constrainers;

import java.awt.Point;

import java.util.ArrayList;

import jdraw.framework.DrawContext;
import jdraw.framework.DrawGrid;
import jdraw.framework.Figure;
import jdraw.framework.FigureHandle;

/**
 * @author Max Rossmannek
 */

public class SnapGrid extends AbstractGrid {

    private static final int SNAP_DIST = 25;

    private Figure current;

    public SnapGrid(DrawContext c) {
        super(c, "Snap Grid", 1, 1);
        this.context.getModel().addModelChangeListener(e -> current = e.getFigure());
    }

    private FigureHandle findNearHandleOf(FigureHandle h) {
        for (Figure f: this.context.getModel().getFigures()) {
            if (!this.context.getView().getSelection().contains(f)) {
                if (f.getHandles() != null) {
                    for (FigureHandle fh: f.getHandles()) {
                        if (h.getLocation().distance(fh.getLocation()) <= SNAP_DIST) {
                            return fh;
                        }
                    }
                }
            }
        }
        return null;
    }

    private Point p0; // snapped mouse coordinates
    private boolean snapped = false;

    private Point oldp;

    public void mouseDown() {
        oldp = null;
    }

    public void mouseUp() { }


    @Override
    public Point constrainPoint(Point p) {
        if (oldp == null) {
            oldp = p;
            return p;
        }

        // existing figure being moved
        if (snapped) {
            oldp = p;
            if (p0.distance(p) <= SNAP_DIST) {
                return p0;
            } else {
                snapped = false;
                return p;
            }
        }

        if (this.context.getView().getSelection().size() > 0) {

            for (Figure f: this.context.getView().getSelection()) {
                if (f.getHandles() != null) {
                    for (FigureHandle fh: f.getHandles()) {
                        FigureHandle near = findNearHandleOf(fh);
                        if (near != null) {
                            snapped = true;
                            int dx = near.getLocation().x - fh.getLocation().x;
                            int dy = near.getLocation().y - fh.getLocation().y;
                            p0 = new Point(oldp.x + dx, oldp.y + dy);
                            oldp = p;
                            return p0;
                        }
                    }
                }
            }

        } else {

            // new figure insertion
            for (Figure f: this.context.getModel().getFigures()) {
                if (f != current) {
                    if (f.getHandles() != null) {
                        for (FigureHandle fh: f.getHandles()) {
                            if (p.distance(fh.getLocation()) <= SNAP_DIST) {
                                return fh.getLocation();
                            }
                        }
                    }
                }
            }

        }
        oldp = p;
        return p;
    }

}
