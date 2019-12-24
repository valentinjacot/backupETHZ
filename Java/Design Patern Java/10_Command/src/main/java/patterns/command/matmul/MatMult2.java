package patterns.command.matmul;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * This class multiplies two N by N matrices.
 */
public final class MatMult2 {
	/** Dimension of the matrices. */
	private static final int N = 1000;
	
	/** MatMult is a utility class, no instances are ever created. */
	private MatMult2() { }

	/** computes one row of the product a*b. */
	static class ComputeRow implements Callable<int[]> {
		private final int row;
		private final int[][] a, b;
		private final int[] result;

		public ComputeRow(int row, int[][] a, int[][] b) { 
			this.row = row; 
			this.a = a; this.b = b;
			this.result = new int[N];
		}
		
		@Override
		public int[] call() {
			for (int i = 0; i < N; i++) {
				int sum = 0;
				for (int j = 0; j < N; j++) {
					sum += a[row][j] * b[j][i];					
				}
				result[i] = sum;
			}
			return result;
		}
	}
	
	public static void main(String[] args) throws Exception {
		int[][] a = new int[N][N];
		int[][] b = new int[N][N];
		int[][] c = new int[N][N];
		
		createRandomMatrix(a);
		createRandomMatrix(b);
		
		System.out.println("starting");
		System.out.println(Runtime.getRuntime().availableProcessors());
		
		long start = System.currentTimeMillis();
		computeProductSequential(a, b, c);
		long end = System.currentTimeMillis();
		System.out.println("seq done [" + (end-start) + " msec]");
		
		int nOfThreads = 1;
		while(nOfThreads <= Runtime.getRuntime().availableProcessors()) {
			start = System.currentTimeMillis();
			computeProductParallel(nOfThreads, a, b, c);
			end = System.currentTimeMillis();
			System.out.printf("par[%d] done [%5d msec]\n", nOfThreads, end-start);
			nOfThreads = 2*nOfThreads;
		}
	}

	/**
	 * Initializes the given matrix with random values. The matrix may be jagged.
	 * @param mat matrix to be defined.
	 */
	private static void createRandomMatrix(final int[][] mat) {
		Random r = new Random();
		for (int i = 0; i < mat.length; i++) {
			for (int j = 0; j < mat[i].length; j++) {
				mat[i][j] = r.nextInt();
			}
		}
	}

	private static void computeProductSequential(final int[][] a, final int[][] b, final int[][] c) {
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				int sum = 0;
				for (int k = 0; k < N; k++) {
					sum += a[i][k] * b[k][j];
				}
				c[i][j] = sum;
			}
		}
	}

	private static void computeProductParallel(int nOfThreads, final int[][] a,
			final int[][] b, final int[][] c) throws InterruptedException, ExecutionException {
		ExecutorService ex = Executors.newFixedThreadPool(nOfThreads);
		List<ComputeRow> tasks = new ArrayList<>(N);
		for (int i = 0; i < N; i++) {
			tasks.add(new ComputeRow(i, a, b));
		}
		List<Future<int[]>> result = ex.invokeAll(tasks);
		for(int i=0; i<N; i++) {
			c[i] = result.get(i).get();
		}
		ex.shutdown();
	}
	
}
