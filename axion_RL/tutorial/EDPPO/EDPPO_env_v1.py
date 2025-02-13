# EDPPO Environment

import numpy as np
import os
import time
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

import meshio
import pygmsh
from dolfin import *

from EDPPO_mesh_util import compute_volume


class SimLabEnv(object):
    def __init__(self):
        # Design Domain _ 모델에따라 설정 _220118
        self.design_length_x = 8.  # Design Domain x축 길이
        self.design_length_y = 4.  # Design Domain y축 길이
        self.unit_num_x = 80
        self.unit_num_y = 40

        # State
        self.observation_space = np.ones((self.unit_num_x, self.unit_num_y, 1))
        self.action_space = np.zeros((self.unit_num_x, self.unit_num_y, 1))

        # Count
        self.step_counter = 0

        np.random.seed(42)

    def reset(self, cpu_id):
        # 디렉토리 설정
        self.set_dir(cpu_id)

        # state, decision_list, step 초기화
        self.observation_space = np.ones((self.unit_num_x, self.unit_num_y, 1))
        self.step_counter = 0

        return self.observation_space

    def set_dir(self, cpu_id):
        # 디렉토리가 없으면 디렉토리 생성
        if not os.path.isdir(f"./FEniCS/Proc_{cpu_id}"):
            Path(f"./FEniCS/Proc_{cpu_id}").mkdir(parents=True, exist_ok=True)

        # FEniCS 해석파일이 저장 되는 디렉토리
        self.proc_dir = f"./FEniCS/Proc_{cpu_id}"
        # 결과, 최적파일 이름
        self.result_dir = f"./FEniCS/Proc_{cpu_id}/Cantilever_results"


    def step(self, action, opt_val, cpu_id):
        # 시간 측정 시작
        start_time = time.time()
        # # for debugging
        # np.savetxt("/home/breakthsj/RLD/220117_EDPPO_FEniCS/action_debug.txt", action)

        reward, vol, opt_val, target_val = self.set_reward(action, opt_val, cpu_id)
        # Done -> True
        # 다단으로 생성 시 z 좌표를 기준으로 done 조건 확인 코드 개발
        done = True
        next_state = action.reshape([self.unit_num_x, self.unit_num_y, 1])

        # 에피소드마다 걸린 시간 측정
        end_time = time.time()
        spend_time = float(end_time - start_time)

        return next_state, reward, done, target_val, opt_val, vol, spend_time


    def set_reward(self, action, opt_val, cpu_id):
        # FEniCS 실행
        # print(f"{cpu_id}번 프로세스 FEniCS 백그라운드 동작 중...")

        # Reward Function
        initial_area = self.design_length_x * self.design_length_y
        target_val, vol = self.structural_solver(action)
        vol_ratio = vol / initial_area

        # absolute value for malfunction
        target_val = np.abs(target_val)

        # 연결이 끊어지거나 하중점, 지지점들이 지워졌을경우
        if target_val <= 0:
            reward = 0
        # constraint: 영역 50%
        elif vol_ratio > 0.5:
            constraint_discount = 5*(vol_ratio-0.5)**0.7 + 1
            reward = np.power(target_val*constraint_discount, -1)
        else:
            reward = np.power(target_val, -1)
            # 현재까지 최적기록 저장
            if opt_val >= target_val:
                opt_val = target_val
                # shutil.copy2(self.result_dir, self.opti_dir+".xdmf")  # 덮어쓰기 가능

        return reward, vol, opt_val, target_val

    def structural_solver(self, action_list):
        # mesh import
        pygmsh_mesh_dir, vol = self.mesh_gen(action_list)
        mesh = Mesh()
        with XDMFFile(pygmsh_mesh_dir) as infile:
            infile.read(mesh)

        # # 디버깅 / Mesh 요소 정보, 생성 시간, 그래픽 확인
        # print("Plotting a RectangleMesh")
        # # print("Mesh time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
        # print(f"Mesh element :{mesh.num_cells()}")  # 메쉬 정보 추출
        # print(f"Mesh Dimension :{mesh.geometric_dimension()}")  # 메쉬 정보 추출
        # plt.figure()
        # plot(mesh, title="Rectangle (right/left)")
        # plt.show()

        # Problem parameters
        E = Constant(1)
        nu = Constant(0.3)
        lamda = E * nu / (1 + nu) / (1 - 2 * nu)
        mu = E / (2 * (1 + nu))
        T = Constant((0, -1))  # vertical downwards force

        # Boundaries
        def left(x, on_boundary):
            return near(x[0], 0.) and on_boundary

        def load(x, on_boundary):
            return near(x[0], self.design_length_x) and near(x[1], self.design_length_y/2., 0.05)

        facets = MeshFunction("size_t", mesh, 1)
        AutoSubDomain(load).mark(facets, 1)
        ds = Measure("ds", subdomain_data=facets)

        # Function space for displacement
        V = VectorFunctionSpace(mesh, "CG", 2)
        # Fixed boundary condtions
        bc = DirichletBC(V, Constant((0, 0)), left)

        def eps(v):
            return sym(grad(v))

        def sigma(v):
            return lamda * div(v) * Identity(2) + 2 * mu * eps(v)

        # Inhomogeneous elastic variational problem
        v = TestFunction(V)
        u_ = TrialFunction(V)
        a = inner(sigma(u_), eps(v)) * dx
        L = dot(T, v) * ds(1)

        u = Function(V, name="Displacement")
        solve(a == L, u, bc)
        compliance = assemble(action(L, u))

        # # 디버깅 / 해석시간, 결과값, 그래픽 확인
        # print("Analysis time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
        # print("compliance =", compliance)
        # plot(u, mode="displacement")
        # plt.show()

        # 응력 공간 구현
        Vsig = TensorFunctionSpace(mesh, "DG", degree=0)
        sig = Function(Vsig, name="Stress")
        sig.assign(project(sigma(u), Vsig))

        # 해석 파일 저장
        # result_dir = os.path.join(self.proc_dir, "Cantilever_results.xdmf")
        file_results = XDMFFile(self.result_dir+'.xdmf')
        file_results.parameters["flush_output"] = True
        file_results.parameters["functions_share_mesh"] = True
        file_results.write(u, 0.)
        file_results.write(sig, 0.)

        return compliance, vol

    def mesh_gen(self, action):
        # Coordinates of lower-left and upper-right vertices of a square domain
        xmin = 0.0
        xmax = self.design_length_x
        ymin = 0.0
        ymax = self.design_length_y
        x_num_cell = self.unit_num_x
        y_num_cell = self.unit_num_y

        with pygmsh.occ.Geometry() as geom:
            # ignore terminal print
            # pygmsh._optimize.gmsh.option.setNumber("General.Terminal", 0)
            pygmsh._optimize.gmsh.option.setNumber("General.Verbosity", 2)

            # mesh 해상도 설정
            geom.characteristic_length_min = 0.05
            geom.characteristic_length_max = 0.05

            X = np.linspace(xmin, xmax, x_num_cell)
            Y = np.linspace(ymin, ymax, y_num_cell)
            X, Y = np.meshgrid(X, Y, indexing='ij')
            Z = np.zeros(x_num_cell * y_num_cell)
            R = np.full(X.size, (xmax - xmin) / x_num_cell)

            design_nodes = np.stack((X.flatten(), Y.flatten()), axis=1)
            design_nodes = np.c_[design_nodes, Z]

            # model base generation
            base = geom.add_rectangle([0.0, 0.0, 0.0], self.design_length_x, self.design_length_y)

            # subtract Circles generation
            circles = [geom.add_disk(design_node, r) for index, (design_node, r) in enumerate(zip(design_nodes, R)) if not action[index]]
            subtract_geo = geom.boolean_union(circles)

            # boolean subtract
            cut = geom.boolean_difference(base, subtract_geo)

            # add physical boundary (Load_essential)
            p0 = geom.add_point([self.design_length_x - 0.1, self.design_length_y / 2.0, 0.], 0.05)
            p1 = geom.add_point([self.design_length_x, self.design_length_y / 2.0 - 0.05, 0.], 0.05)
            p2 = geom.add_point([self.design_length_x, self.design_length_y / 2.0 + 0.05, 0.], 0.05)
            line_1 = geom.add_line(p0, p1)
            load_line = geom.add_line(p1, p2)
            line_2 = geom.add_line(p2, p0)
            # geom.add_physical(load_line, label="load")
            lineloop = geom.add_curve_loop([line_1, load_line, line_2])
            load_geo = geom.add_plane_surface(lineloop)

            geom.boolean_union([cut, load_geo])

            mesh = geom.generate_mesh(dim=2)

        # cells = np.vstack(np.array([cells.data for cells in mesh.cells if cells.type == "triangle"]))
        # triangle_mesh = meshio.Mesh(points=mesh.points[:, :2], cells=[("triangle", cells)])

        # 2D mesh
        points = mesh.points[:, :2]
        cells_ = mesh.get_cells_type("triangle")
        triangle_mesh = meshio.Mesh(points=points, cells={"triangle": cells_})

        vol = compute_volume(triangle_mesh)

        # Write mesh
        pygmsh_mesh_dir = os.path.join(self.proc_dir, "Cantilever_mesh.xdmf")
        meshio.xdmf.write(pygmsh_mesh_dir, triangle_mesh)

        return pygmsh_mesh_dir, vol