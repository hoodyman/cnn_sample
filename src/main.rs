type FType = f64;

fn main() {
    let x = vec![
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        vec![0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0],
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0],
    ];

    let kernel = vec![
        vec![1.0, -1.0, -1.0],
        vec![-1.0, 1.0, -1.0],
        vec![-1.0, -1.0, 1.0],
    ];

    #[inline(always)]
    fn value_or_zero(i: i32, j: i32, a: &Vec<Vec<FType>>) -> FType {
        if i < 0 || j < 0 || i > (a.len() - 1) as i32 || j > (a[0].len() - 1) as i32 {
            return 0.0;
        }
        a[i as usize][j as usize]
    }

    #[inline(always)]
    fn fold_sample(
        sample_grid: &Vec<Vec<FType>>,
        sample_grid_i: usize,
        sample_grid_j: usize,
        kernel: &Vec<Vec<FType>>,
    ) -> FType {
        let kernel_width = kernel[0].len();
        let kernel_height = kernel.len();

        assert_eq!(kernel_width % 2 == 1 && kernel_height % 2 == 1, true);

        let i_offset = kernel_height / 2;
        let j_offset = kernel_width / 2;

        let mut sum = 0.0;

        for i in 0..kernel_height {
            for j in 0..kernel_width {
                sum += kernel[i][j]
                    * value_or_zero(
                        i as i32 - i_offset as i32 + sample_grid_i as i32,
                        j as i32 - j_offset as i32 + sample_grid_j as i32,
                        &sample_grid,
                    );
            }
        }

        if sum > 0.0 {
            sum
        } else {
            0.0
        }
    }

    fn fold(input: &Vec<Vec<FType>>, kernel: &Vec<Vec<FType>>, stride: usize) -> Vec<Vec<FType>> {
        let mut output_width = input[0].len() / stride;
        if input[0].len() % stride != 0 {
            output_width += 1;
        }
        let mut output_height = input.len() / stride;
        if input.len() % stride != 0 {
            output_height += 1;
        }
        let mut output = vec![vec![0.0; output_width]; output_height];
        for i in (0..input.len()).step_by(stride) {
            for j in (0..input[0].len()).step_by(stride) {
                let value = fold_sample(input, i, j, kernel);
                output[i / stride][j / stride] = value;
            }
        }
        output
    }

    fn pool(input: &Vec<Vec<FType>>, size: usize) -> Vec<Vec<FType>> {
        let mut output_width = input[0].len() / size;
        if input[0].len() % size != 0 {
            output_width += 1;
        }
        let mut output_height = input.len() / size;
        if input.len() % size != 0 {
            output_height += 1;
        }
        let mut output = vec![vec![0.0; output_width]; output_height];
        for i in (0..input.len()).step_by(size) {
            for j in (0..input[0].len()).step_by(size) {
                let mut sum = 0.0;
                for i2 in i..i + size {
                    for j2 in j..j + size {
                        if i2 < input.len() && j2 < input[0].len() {
                            sum += input[i2][j2];
                        }
                    }
                }
                output[i / size][j / size] = sum / (size as FType).powf(2.0);
            }
        }
        output
    }

    println!("START");
    for i in 0..x.len() {
        println!("{:10.3?}", x[i]);
    }
    println!("FOLD");

    let f = fold(&x, &kernel, 1);
    for i in 0..f.len() {
        println!("{:10.3?}", f[i]);
    }
    println!("POOL");

    let p = pool(&f, 2);
    for i in 0..p.len() {
        println!("{:10.3?}", p[i]);
    }
    println!("=====");
}
