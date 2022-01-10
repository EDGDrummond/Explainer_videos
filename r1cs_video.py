#Manim Community v0.14.0

import inspect

from manim import *
from manim.utils.rate_functions import ease_out_bounce, ease_in_bounce, ease_out_sine, there_and_back, ease_in_out_sine
from manim_physics import *

# Adapted from the Charge class in manim-physics
class DotGlow(VGroup):

    def __init__(self, magnitude=1, point=ORIGIN, color1=RED_D, color2=RED_A, **kwargs):
        VGroup.__init__(self, **kwargs)

        layer_num = 80
        color_list = color_gradient([color1, color2], layer_num)
        opacity_func = lambda t: 1500 * (1 - abs(t - 0.009) ** 0.0001)
        rate_func = lambda t: t ** 2

        for i in range(layer_num):
            self.add(
                Arc(
                    radius=magnitude * rate_func((0.5 + i) / layer_num),
                    angle=TAU,
                    color=color_list[i],
                    stroke_width=101
                    * (rate_func((i + 1) / layer_num) - rate_func(i / layer_num))
                    * magnitude,
                    stroke_opacity=opacity_func(rate_func(i / layer_num)),
                ).shift(point)
            )

class Glow(VGroup):
    def __init__(self, mobject, magnitude=1, color1=RED_D, color2=RED_A, **kwargs):
        VGroup.__init__(self, **kwargs)

        center = mobject.get_center()
        layer_num = 80
        color_list = color_gradient([color1, color2], layer_num)
        opacity_func = lambda t: 1500 * (1 - abs(t - 0.009) ** 0.0001)
        rate_func = lambda t: t ** 2

        for i in range(layer_num):
            self.add(
                mobject.copy()
                # .scale(magnitude * i)
                .set_color(color_list[i])
                .set_stroke(
                    width=101 * (rate_func((i + 1) / layer_num) - rate_func(i / layer_num)) * magnitude,
                    opacity=opacity_func(rate_func(i / layer_num)))
                .shift(center)
            )

from shapely import geometry as gm
def intersection(vmob1: VMobject, vmob2: VMobject):
    """intersection points of 2 curves"""
    a = gm.LineString(vmob1.points)
    b = gm.LineString(vmob2.points)
    intersects: gm.GeometryCollection = a.intersection(b)
    try:  # for intersections > 1
        return np.array(
            [[[x, y, z] for x, y, z in m.coords][0] for m in intersects.geoms]
        )
    except:  # else
        return np.array([[x, y, z] for x, y, z in intersects.coords])

class Video(MovingCameraScene):
    def construct(self):
        BG = "#12141d"
        DARK = "#154bf9"
        LIGHT = "#00c0f9"
        RED = "#ff6f5f"
        PINK = "#e561e5"
        PURPLE = "#931cff"
        BLACK = "#000000"
        WHITE = "#ffffff"
        
        self.camera.background_color = BLACK
        # self.camera.frame.scale(4)

        axes = Axes([-6, 10], [-8, 8]).set_color(WHITE)
        
        self.play(FadeIn(axes))
        self.wait()
        
        # axes.y_axis.submobjects[0].set_y(axes.y_axis.submobjects[0].get_bottom()[1] - axes.y_axis.submobjects[1][0].get_top()[1])
        # print(axes.y_axis.submobjects)
        
        EC_generic_text = MathTex("y^{2} = x^{3} + Ax + B").to_corner(UR)
        curve_text = MathTex("y^{2} = x^{3} - 3x + 3").next_to(EC_generic_text, DOWN)
        underline = Line(curve_text.get_left(), curve_text.get_right(), color=LIGHT, stroke_width=0.75).next_to(curve_text, DOWN, buff=SMALL_BUFF)

        graph = axes.plot_implicit_curve(
            lambda x, y: x**3 - 3*x + 3 - y**2,
            color=PURPLE
        )
        # self.add(Glow(graph, 0.3))
        # self.add(graph)
        self.play(Write(EC_generic_text, run_time = 1.5))
        self.wait()
        self.play(
            # FadeIn(graph),#, shift=5*RIGHT, rate_func=ease_out_sine, run_time=0.75),
            Write(curve_text, run_time=1.5),
            # GrowFromEdge(underline, LEFT, point_color=WHITE)
            # Write(graph)
        )
        self.play(
            Circumscribe(curve_text, color=LIGHT)
            # GrowFromEdge(underline, LEFT, rate_func=rush_from)
        )
        self.play(
            FadeIn(graph)
        )
        # self.make_static_body(graph)#Line(6*LEFT, 6*RIGHT).shift(4*DOWN))
        self.wait()
        self.play(FadeOut(EC_generic_text))
        self.wait()
        self.play(curve_text.animate.shift(UP), run_time = 1)
        self.wait()

        coord1 = [-2, 1]
        coord2 = [-1.29, 2.17, 0]
        R = Dot(point=axes.c2p(*coord1), color=WHITE)
        # test_square = Square()
        R_glow = DotGlow(magnitude=0.3, point=R.get_center(), color1=LIGHT, color2=DARK)
        R_glow_weak = DotGlow(magnitude=0.1, point=R.get_center(), color1=LIGHT, color2=DARK)
        # R_glow.add_updater(
        #     lambda g: g.move_to(R.get_center())
        # )

        R2 = Dot(point=axes.c2p(*coord2), color=WHITE)
        R2_glow_weak = DotGlow(magnitude=0.1, point=R2.get_center(), color1=LIGHT, color2=DARK)
        R2_glow = DotGlow(magnitude=0.3, point=R2.get_center(), color1=LIGHT, color2=DARK)
        
        axes.add_coordinates([coord1[0], coord1[0]], [coord1[1], coord1[1]])
        labels = axes.coordinate_labels
        # axes.remove(labels)
        # self.wait()
        
        bg_recs = VGroup(*[BackgroundRectangle(label, color=BG, buff=0.05) for label in labels]).set_z_index(10)
        axes.coordinate_labels.set_z_index(11)

        # self.add(R)#, test_square)
        # self.make_rigid_body(R)
        self.play(
            FadeIn(R),
            # FadeIn(labels),
            FadeIn(bg_recs),
            FadeIn(R_glow)
        )

        # self.interactive_embed()
        self.wait()

        axes.get_x_axis().remove(axes.get_x_axis().submobjects[-1])
        axes.get_y_axis().remove(axes.get_y_axis().submobjects[-1])
        self.remove(bg_recs)

        self.play(
            FadeOut(R, rate_func=ease_out_bounce),
            Transform(R_glow, R_glow_weak, rate_func=ease_out_bounce, remover=True)
            # MoveAlongPath(R, graph)
        )
        # self.play(
        #     FadeIn(R2, rate_func=ease_in_bounce),
        #     ReplacementTransform(R2_glow_weak, R2_glow, rate_func=ease_in_bounce)
        # )
        # self.remove(R2_glow)
        # self.camera.frame.move_to(R2).scale(0.3)
        tangent = TangentLine(graph, 0.59, length=9).move_to(R2)#Line(LEFT, RIGHT)

        # tangent.rotate(angle_of_tangent(coord2[0], coord2[1], -1.28, 2.178)).move_to(R2.get_center())
        # print(axes.slope_of_tangent(-1.29, graph))

        # for x in [x/10000 for x in range(5000, 10000)]:
        #     print(x, graph.point_from_proportion(x), graph.proportion_from_point(graph.point_from_proportion(x)))

        # Adding a point to itself
        P_inter = Dot(intersection(graph, tangent)[-1])
        Ptimes2 = P_inter.copy().set_y(-P_inter.get_y())
        self_add = MathTex("P + P = 2P").next_to(EC_generic_text, DOWN)

        self.play(FadeIn(R2), Write(self_add, run_time = 1.5))
        self.play(FadeIn(tangent))
        self.wait()
        self.play(FadeIn(P_inter))
        self.wait()
        self.play(FadeOut(tangent))

        # P_2P = Line(P_inter.copy().set_y(P_inter.get_y()+2), Ptimes2.copy().set_y(Ptimes2.get_y()-2))
        Pinter_2P = Line(P_inter, Ptimes2.copy().set_y(Ptimes2.get_y())).set_length(6)
        self.play(FadeIn(Pinter_2P))
        self.wait()
        self.play(FadeIn(Ptimes2))
        self.wait()
        self.play(FadeOut(Pinter_2P))
        self.wait()
        self.play(FadeOut(P_inter))
        self.wait()
        self.play(FadeOut(self_add))

        # Adding different points
        P_2P = Line(R2, Ptimes2.copy().set_y(Ptimes2.get_y())).set_length(7)
        diff_add = MathTex("P + 2P = 3P").next_to(EC_generic_text, DOWN)

        self.play(FadeIn(P_2P), Write(diff_add, run_time = 1.5))
        self.wait()

        P3_inter = Dot(intersection(graph, P_2P)[1])

        self.play(FadeIn(P3_inter))
        self.wait()
        self.play(FadeOut(P_2P))
        self.wait()

        P3 = P3_inter.copy().set_y(-P3_inter.get_y())
        P3inter_3P = Line(P3_inter, P3).set_length(6)
        self.play(FadeIn(P3inter_3P))
        self.wait()
        self.play(FadeIn(P3))
        self.wait()
        self.play(FadeOut(P3inter_3P), FadeOut(P3_inter))
        self.wait()
        self.play(FadeOut(P3), FadeOut(Ptimes2), FadeOut(diff_add))
        self.wait()
        
        # Identity point
        identity_equ = MathTex("P + I = P").next_to(EC_generic_text, DOWN)
        self.play(Write(identity_equ, run_time = 1.5))
        self.wait()

        minus_P = R2.copy().set_y(-R2.get_y())
        self.play(FadeIn(minus_P))
        self.wait()
        P_ident = Line(R2, minus_P).set_length(8)
        self.play(FadeIn(P_ident))


class Infinity(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BLACK
        # self.camera.frame.scale(4)

        axes = Axes([-4, 4], [-4, 4], x_length = 4, y_length = 4, tips = True).set_color(WHITE)
        graph = axes.plot_implicit_curve(
            lambda x, y: x**3 - 3*x + 3 - y**2,
            color=PURPLE
        )
        
        self.play(
            # FadeIn(graph),
            FadeIn(axes),
        )
        self.wait()

        circle = Circle(radius=3, color=WHITE)
        self.play(FadeIn(circle))
        self.wait()



        
class Computation(Scene):
    def construct(self):
        self.camera.background_color = BLACK
        LIGHT = "#00c0f9"
        DARK = "#154bf9"

        # Introduce the equation we are computing, and the separate steps
        equation = MathTex("x^{3} + x + 5 = 35", color=YELLOW)
        steps = MathTex(r'{{\operatorname{int} &= x*x}} \\ {{\operatorname{int}_2 &= \operatorname{int}*x}} \\ {{\operatorname{int}_3 &= \operatorname{int}_2 + x}} \\ {{\operatorname{out} &= \operatorname{int}_3 + 5}}')
        ops = MathTex(r'y &= x \\ y &= x \operatorname{(op)} z')
        vector_compact_1 = MathTex("(a_1, b_1, c_1)")
        vector_compact_2 = MathTex("(a_2, b_2, c_2)")
        vector_compact_3 = MathTex("(a_3, b_3, c_3)")
        vector_compact_4 = MathTex("(a_4, b_4, c_4)").shift(5.2*RIGHT + 1.2*DOWN)
        vector_compact_i = MathTex("(a_i, b_i, c_i)")
        vector_dots = MathTex("\\vdots").shift(5.2*RIGHT + DOWN)
        vector_compact_n = MathTex("(a_n, b_n, c_n)")
        A = Matrix([["a_{11}"], ["\\vdots"], ["a_{1m}"]])
        B = Matrix([["b_{11}"], ["\\vdots"], ["b_{1m}"]])
        C = Matrix([["c_{11}"], ["\\vdots"], ["c_{1m}"]])
        vectors = Group(A, B, C).arrange()
        items = MathTex("{{(}} {{1,\hspace{2mm}}} {{x,\hspace{2mm}}} {{\operatorname{int},\hspace{2mm}}} {{\operatorname{int}_2,\hspace{2mm}}} {{\operatorname{int}_3,\hspace{2mm}}} {{\operatorname{out}}} {{)}}").shift(3.5*DOWN)
        AA = Matrix([["a_{11}"], ["a_{12}"], ["a_{13}"], ["a_{14}"], ["a_{15}"], ["a_{16}"]])
        BB = Matrix([["b_{11}"], ["b_{12}"], ["b_{13}"], ["b_{14}"], ["b_{15}"], ["b_{16}"]])
        CC = Matrix([["c_{11}"], ["c_{12}"], ["c_{13}"], ["c_{14}"], ["c_{15}"], ["c_{16}"]])
        vectorss = Group(AA, BB, CC).arrange()
        S = Matrix([[1], ["x"], ["\operatorname{int}"], ["\operatorname{int}_2"], ["\operatorname{int}_3"], ["\operatorname{out}"]])
        S2 = Matrix([[1], ["x"], ["\operatorname{int}"], ["\operatorname{int}_2"], ["\operatorname{int}_3"], ["\operatorname{out}"]])
        S3 = Matrix([[1], ["x"], ["\operatorname{int}"], ["\operatorname{int}_2"], ["\operatorname{int}_3"], ["\operatorname{out}"]])
        AA = Matrix([["a_{11}"], ["a_{12}"], ["a_{13}"], ["a_{14}"], ["a_{15}"], ["a_{16}"]])
        BB = Matrix([["b_{11}"], ["b_{12}"], ["b_{13}"], ["b_{14}"], ["b_{15}"], ["b_{16}"]])
        CC = Matrix([["c_{11}"], ["c_{12}"], ["c_{13}"], ["c_{14}"], ["c_{15}"], ["c_{16}"]])
        times = Tex("*")
        equals = Tex("=")
        togeth = Group(S, AA, times, S2, BB, equals, S3, CC).arrange()
        parts = togeth.split()
        rect = Rectangle(width = 2.5, height = 0.6, color = LIGHT).shift(3.8*LEFT + 2*UP)
        rect2 = Rectangle(width = 2.5, height = 0.6, color = LIGHT).shift(3.8*LEFT + 1.2*UP)
        rect3 = Rectangle(width = 2.5, height = 0.6, color  = LIGHT).shift(3.8*LEFT + 1.2*DOWN)
        rect4 = Rectangle(width = 2.5, height = 0.6, color  = LIGHT).shift(2*UP)
        rect5 = Rectangle(width = 2.5, height = 0.6, color  = LIGHT).shift(4*RIGHT + 2*DOWN)
        rect6 = Rectangle(width = 2.5, height = 0.6, color  = LIGHT).shift(3.8*LEFT + 0.4*DOWN)
        rect7 = Rectangle(width = 2.5, height = 0.6, color  = LIGHT).shift(2*UP)
        rect8 = Rectangle(width = 2.5, height = 0.6, color  = LIGHT).shift(4*RIGHT + 1.2*DOWN)
        equals_clone = equals.copy().move_to(times.get_center())
        summ = MathTex(r'{{1*a_{11}}}{{+x*a_{12}}}{{+\operatorname{int}*a_{13} \\ +\operatorname{int}_2*a_{14}+\operatorname{int}_3*a_{15}\\ +\operatorname{out}*a_{16}}}').shift(RIGHT)
        AAA = Matrix([["0"], ["1"], ["0"], ["0"], ["0"], ["0"]]).set_color(BLUE).move_to(AA.get_center())
        BBB = Matrix([["0"], ["1"], ["0"], ["0"], ["0"], ["0"]]).set_color(BLUE).move_to(BB.get_center())
        CCC = Matrix([["0"], ["0"], ["1"], ["0"], ["0"], ["0"]]).set_color(BLUE).move_to(CC.get_center())
        AAA2 = Matrix([["0"], ["0"], ["1"], ["0"], ["0"], ["0"]]).set_color(GREEN).move_to(AA.get_center())
        BBB2 = Matrix([["0"], ["1"], ["0"], ["0"], ["0"], ["0"]]).set_color(GREEN).move_to(BB.get_center())
        CCC2 = Matrix([["0"], ["0"], ["0"], ["1"], ["0"], ["0"]]).set_color(GREEN).move_to(CC.get_center())
        AAA3 = Matrix([["0"], ["1"], ["0"], ["2"], ["0"], ["0"]]).set_color(RED).move_to(AA.get_center())
        BBB3 = Matrix([["1"], ["0"], ["0"], ["0"], ["0"], ["0"]]).set_color(RED).move_to(BB.get_center())
        CCC3 = Matrix([["0"], ["0"], ["0"], ["0"], ["1"], ["0"]]).set_color(RED).move_to(CC.get_center())
        AAA4 = Matrix([["5"], ["0"], ["0"], ["0"], ["1"], ["0"]]).set_color(PINK).move_to(AA.get_center())
        BBB4 = Matrix([["1"], ["0"], ["0"], ["0"], ["0"], ["0"]]).set_color(PINK).move_to(BB.get_center())
        CCC4 = Matrix([["0"], ["0"], ["0"], ["0"], ["0"], ["1"]]).set_color(PINK).move_to(CC.get_center())
        fakeS = Matrix([["1"], ["1"], ["1"], ["1"], ["1"], ["6"]]).set_color(DARK).move_to(S.get_center())
        fakeS2 = Matrix([["1"], ["1"], ["1"], ["1"], ["1"], ["6"]]).set_color(DARK).move_to(S2.get_center())
        fakeS3 = Matrix([["1"], ["1"], ["1"], ["1"], ["1"], ["6"]]).set_color(DARK).move_to(S3.get_center())
        fake_sum = MathTex(r'{{(}}{{5}} {{+}}{{1}}{{)*(}}{{1}}{{)=}}{{6}}').shift(3.5*DOWN)
        fake_sum2 = MathTex(r'{{(}}{{1}} {{+}}{{2}}{{)*(}}{{1}}{{)}}{{=}}{{1}}').shift(2.9*DOWN)
        not_equals = MathTex("\\neq").move_to(fake_sum2[8].get_center()).set_color(RED)
        SS = Matrix([["1"], ["3"], ["9"], ["27"], ["30"], ["35"]]).set_color(YELLOW).move_to(S.get_center())
        SS2 = Matrix([["1"], ["3"], ["9"], ["27"], ["30"], ["35"]]).set_color(YELLOW).move_to(S2.get_center())
        SS3 = Matrix([["1"], ["3"], ["9"], ["27"], ["30"], ["35"]]).set_color(YELLOW).move_to(S3.get_center())
        
        
        self.introduce_constraints(equation, steps)
        self.allowed_ops(ops, LIGHT, steps)
        self.introduce_R1CS(vector_compact_1, vector_compact_2, vector_compact_3, vector_compact_4, vector_compact_i, vector_dots, vector_compact_n, A, B, C, vectors)
        self.vector_size(items, LIGHT, vectors, vectorss, steps, vector_compact_1, vector_compact_2, vector_compact_3, vector_compact_4)
        self.write_dot_prod(items, parts, rect, rect2, equals_clone, summ, vectorss)
        self.instantiating_values(AA, BB, CC, AAA, BBB, CCC, AAA2, BBB2, CCC2, AAA3, BBB3, CCC3, AAA4, BBB4, CCC4, steps)
        self.fake_sol(fakeS, fakeS2, fakeS3, S, S2, S3, rect, rect3, rect4, rect5, fake_sum)
        self.fake_sol_2(steps, AAA3, AAA4, BBB3, BBB4, CCC3, CCC4, rect2, rect6, rect7, rect8, fake_sum2, not_equals)
        self.actual_sol(fake_sum, fake_sum2, not_equals, fakeS, fakeS2, fakeS3, SS, SS2, SS3, AA, BB, CC, AAA3, BBB3, CCC3, parts, steps, AAA, BBB, CCC, AAA2, BBB2, CCC2, AAA4, BBB4, CCC4, equation, LIGHT)

    def introduce_constraints(self, equation, steps):
        self.play(Write(equation, run_time = 1))
        self.wait
        self.play(equation.animate.shift(3.5*UP))
        self.wait
        self.play(FadeIn(steps[0]), run_time=1)
        self.wait
        self.play(FadeIn(steps[2]), run_time=1)
        self.wait
        self.play(FadeIn(steps[4]), run_time=1)
        self.wait
        self.play(FadeIn(steps[6]), run_time=1)
        self.wait
        self.play(steps.animate.shift(5*LEFT), run_time=1)
        self.wait
        self.play(steps[0].animate.set_color(BLUE),
        steps[2].animate.set_color(GREEN),
        steps[4].animate.set_color(RED),
        steps[6].animate.set_color(PINK),
        )
        self.wait()
        
    def allowed_ops(self, ops, LIGHT, steps):
        # Introduce the allowed operations
        self.play(FadeIn(ops))
        self.wait
        self.play(Circumscribe(steps[0], color = LIGHT), Circumscribe(steps[2], color=LIGHT))
        self.wait
        self.play(Circumscribe(steps[4], color = LIGHT), Circumscribe(steps[6], color=LIGHT))
        self.wait
        self.play(FadeOut(ops))
        self.wait

    def introduce_R1CS(self, vector_compact_1, vector_compact_2, vector_compact_3, vector_compact_4, vector_compact_i, vector_dots, vector_compact_n, A, B, C, vectors):
        # Introduce an R1CS

        self.play(FadeIn(vector_compact_1))
        self.wait
        self.play(FadeOut(vector_compact_1), FadeIn(vectors))
        self.wait
        self.play(FadeOut(vectors), FadeIn(vector_compact_1))
        self.wait
        self.play(vector_compact_1.animate.shift(5.2*RIGHT + 2*UP))
        self.play(FadeIn(vector_compact_2))
        self.wait
        self.play(vector_compact_2.animate.shift(5.2*RIGHT + UP))
        self.play(FadeIn(vector_compact_3))
        self.wait
        self.play(vector_compact_3.animate.shift(5.2*RIGHT))
        self.play(FadeIn(vector_compact_i))
        self.wait
        self.play(FadeTransform(vector_compact_i, vector_dots))
        self.play(FadeIn(vector_compact_n))
        self.wait
        self.play(vector_compact_n.animate.shift(5.2*RIGHT + 2*DOWN))
        self.wait

        # We are going to have 4 sets of these vectors because we have 4 constraints
        self.play(vector_compact_1.animate.shift(0.8*DOWN),
        vector_compact_2.animate.shift(0.6*DOWN),
        vector_compact_3.animate.shift(0.4*DOWN),
        FadeOut(vector_dots),
        FadeTransform(vector_compact_n, vector_compact_4))
        self.wait
        
        vector_compact_1_clone = vector_compact_1.copy()
        self.play(vector_compact_1_clone.animate.shift(5.2*LEFT + 1.2*DOWN))
        self.wait
        self.play(FadeOut(vector_compact_1_clone), FadeIn(vectors))
        self.wait
        
    def vector_size(self, items, LIGHT, vectors, vectorss, steps, vector_compact_1, vector_compact_2, vector_compact_3, vector_compact_4):
        # How many items in each vector?
        self.play(FadeIn(items[0], items[2]))
        self.wait
        self.play(FadeIn(items[4]))
        self.wait
        self.play(FadeIn(items[6]))
        self.wait
        self.play(FadeIn(items[8]))
        self.wait
        self.play(FadeIn(items[10]))
        self.wait
        self.play(FadeIn(items[12], items[14]))
        self.wait

        self.play(*[FadeTransform(vectors[s], vectorss[s]) for s in range(3)])
        self.wait
        self.play(Circumscribe(steps[0], color = LIGHT))
        self.wait
        self.play(FadeOut(steps[2], steps[4], steps[6], vector_compact_1, vector_compact_2, vector_compact_3, vector_compact_4), steps[0].animate.shift(2.3*UP))
        self.wait

    def write_dot_prod(self, items, parts, rect, rect2, equals_clone, summ, vectorss):
        self.play(FadeIn(parts[0]), FadeOut(items), FadeTransform(vectorss[0], parts[1]), FadeOut(vectorss[1], vectorss[2]))
        self.wait()

        # aligning vectors
        self.play(Write(equals_clone))
        self.wait()
        self.play(Write(rect))
        self.wait()
        self.play(FadeIn(summ[0]))
        self.wait()
        self.play(Write(rect2))
        self.wait()
        self.play(FadeIn(summ[2]))
        self.wait()
        self.play(FadeIn(summ[4]))
        self.wait()
        self.play(FadeOut(summ, equals_clone, rect, rect2))
        self.wait()
        self.play(FadeIn(parts[2], parts[3], parts[4]))
        self.wait()
        self.play(FadeIn(parts[5], parts[6], parts[7]))
        self.wait()

    def instantiating_values(self, AA, BB, CC, AAA, BBB, CCC, AAA2, BBB2, CCC2, AAA3, BBB3, CCC3, AAA4, BBB4, CCC4, steps):
        # Instantiating actual values for the sets of vectors
        self.play(FadeTransform(AA, AAA), FadeTransform(BB, BBB), FadeTransform(CC, CCC))
        self.wait()
        steps[2].move_to(steps[0].get_center())
        self.play(FadeTransform(steps[0], steps[2]), FadeTransform(AAA, AAA2), FadeTransform(BBB, BBB2), FadeTransform(CCC, CCC2))
        self.wait()
        steps[4].move_to(steps[0].get_center())
        self.play(FadeTransform(steps[2], steps[4]), FadeTransform(AAA2, AAA3), FadeTransform(BBB2, BBB3), FadeTransform(CCC2, CCC3))
        self.wait()
        steps[6].move_to(steps[0].get_center())
        self.play(FadeTransform(steps[4], steps[6]), FadeTransform(AAA3, AAA4), FadeTransform(BBB3, BBB4), FadeTransform(CCC3, CCC4))
        self.wait()

    def fake_sol(self, fakeS, fakeS2, fakeS3, S, S2, S3, rect, rect3, rect4, rect5, fake_sum):
        # Attempted solution vector
        self.play(FadeTransform(S, fakeS), FadeTransform(S2, fakeS2), FadeTransform(S3, fakeS3))
        self.wait()

        # numberplane = NumberPlane()
        # self.add(numberplane)
        self.wait()
        self.play(Write(rect), Write(rect3), Write(rect4), Write(rect5))
        self.wait
        unfaded = [0,2,3,5,7]
        self.play(FadeTransform(rect, fake_sum[1]), FadeTransform(rect3, fake_sum[4]), FadeTransform(rect4, fake_sum[6]), FadeTransform(rect5, fake_sum[8]), *[FadeIn(fake_sum[s]) for s in unfaded])
        self.wait()

    def fake_sol_2(self, steps, AAA3, AAA4, BBB3, BBB4, CCC3, CCC4, rect2, rect6, rect7, rect8, fake_sum2, not_equals):
        self.play(FadeTransform(AAA4, AAA3), FadeTransform(BBB4, BBB3), FadeTransform(CCC4, CCC3), FadeTransform(steps[6], steps[4]))
        self.wait()
        self.play(Write(rect2), Write(rect6), Write(rect7), Write(rect8))
        self.wait()
        unfaded = [0,2,3,5,7,8]
        self.play(FadeTransform(rect2, fake_sum2[1]), FadeTransform(rect6, fake_sum2[4]), FadeTransform(rect7, fake_sum2[6]), FadeTransform(rect8, fake_sum2[9]), *[FadeIn(fake_sum2[s]) for s in unfaded])
        self.wait()
        self.play(FadeTransform(fake_sum2[8], not_equals))
        self.wait()

    def actual_sol(self, fake_sum, fake_sum2, not_equals, fakeS, fakeS2, fakeS3, SS, SS2, SS3, AA, BB, CC, AAA3, BBB3, CCC3, parts, steps, AAA, BBB, CCC, AAA2, BBB2, CCC2, AAA4, BBB4, CCC4, equation, LIGHT):
        self.play(FadeTransform(fakeS, SS), FadeTransform(fakeS2, SS2), FadeTransform(fakeS3, SS3), FadeTransform(AAA3, AA), FadeTransform(BBB3, BB), FadeTransform(CCC3, CC), FadeOut(fake_sum, fake_sum2, not_equals))
        self.wait()
        self.play(SS.animate.shift(LEFT), FadeOut(SS2, SS3, parts[2], parts[5], AA, BB, CC, steps[4]))
        self.wait()
        constraints = Group(steps[0], steps[2], steps[4], steps[6]).arrange_in_grid(buff=1.5).move_to(2.7*RIGHT)
        self.play(FadeIn(constraints))
        self.wait()

        constraint1 = Group(AAA, BBB, CCC).arrange().scale(0.7)
        constraint2 = Group(AAA2, BBB2, CCC2).arrange().scale(0.7)
        constraint3 = Group(AAA3, BBB3, CCC3).arrange().scale(0.7)
        constraint4 = Group(AAA4, BBB4, CCC4).arrange().scale(0.7)
        constraints_real = Group(constraint1, constraint2, constraint3, constraint4).arrange_in_grid().move_to(3*RIGHT)
        self.play(*[FadeTransform(constraints[s], constraints_real[s]) for s in range(4)], equation.animate.shift(2*LEFT))
        self.wait()

        # Public knowledge
        pub_param_1 = SurroundingRectangle(constraints_real, color = LIGHT)
        pub_param_2 = SurroundingRectangle(SS[0][5], color = LIGHT)
        initial_equ = SurroundingRectangle(equation, color = LIGHT)
        zero_knowledge = SurroundingRectangle(SS[0][0:5], color = PURPLE)
        self.play(Write(pub_param_1))
        self.wait()
        self.play(Write(pub_param_2))
        self.wait()
        self.play(Write(initial_equ))
        self.wait()
        self.play(Write(zero_knowledge))
        self.wait()
        self.play(zero_knowledge.animate.set_fill(PURPLE, opacity=1.0))
        self.wait()
        generic_equ = MathTex("x^{3} + x + 5 = y", color=YELLOW).move_to(equation)
        self.play(pub_param_2.animate.set_fill(LIGHT, opacity=1.0), FadeTransform(equation, generic_equ))
        self.wait()
        self.play(pub_param_2.animate.set_fill(LIGHT, opacity=0), FadeTransform(generic_equ, equation))
        self.wait()



def langrage_basis(x, s):
    result = 1
    for t in range(4):
            if t != s:
                result *= (x - (t+1))/((s+1) - (t+1))
    return result

class Lagrange(Scene):
    def construct(self):
        self.camera.background_color = BLACK
        LIGHT = "#00c0f9"
        DARK = "#154bf9"

        lagrange_axes = Axes(x_range=[0,8],y_range=[0,6], y_length=5, tips=False, axis_config={"include_numbers": True}).shift(0.8*UP)
        vec = [[1, 3, 5, 1]]
        vecc = Matrix(vec).set_color(BLUE).shift(3.5*DOWN + 2.3*LEFT)
        points = [Dot(lagrange_axes.coords_to_point(s+1,vec[0][s]), color=BLUE) for s in range(4)]
        lagrange_bases1 = lambda x: langrage_basis(x, 0)
        lagrange_bases2 = lambda x: langrage_basis(x, 1)
        lagrange_bases3 = lambda x: langrage_basis(x, 2)
        lagrange_bases4 = lambda x: langrage_basis(x, 3)
        lagrangey = [lagrange_bases1, lagrange_bases2, lagrange_bases3, lagrange_bases4]
        colors = [RED, ORANGE, YELLOW, GREEN]
        plots = [*[lagrange_axes.plot(lagrangey[i], color = colors[i]) for i in range(4)]]
        interpoly = lambda x: vec[0][0]*langrage_basis(x, 0)+vec[0][1]*langrage_basis(x, 1)+vec[0][2]*langrage_basis(x, 2)+vec[0][3]*langrage_basis(x, 3)
        interpoly_plot = lagrange_axes.plot(interpoly, color = BLUE)
        interpoly_equ = MathTex("-x^3 + 6x^2 - 9x + 5").set_color(BLUE).shift(3*UP+1.8*LEFT)
        interpo_point = Dot(lagrange_axes.coords_to_point(0,5), color = RED)
        vecc_basis_2 = lambda x: vec[0][1]*langrage_basis(x, 1)
        vecc_basis_3 = lambda x: vec[0][2]*langrage_basis(x, 2)
        vecc_plots = [lagrange_axes.plot(vecc_basis_2, color = colors[1]), lagrange_axes.plot(vecc_basis_3, color = colors[2])]

        AAA = Matrix([["0"], ["1"], ["0"], ["0"], ["0"], ["0"]]).set_color(BLUE)
        BBB = Matrix([["0"], ["1"], ["0"], ["0"], ["0"], ["0"]]).set_color(BLUE)
        CCC = Matrix([["0"], ["0"], ["1"], ["0"], ["0"], ["0"]]).set_color(BLUE)
        AAA2 = Matrix([["0"], ["0"], ["1"], ["0"], ["0"], ["0"]]).set_color(GREEN)
        BBB2 = Matrix([["0"], ["1"], ["0"], ["0"], ["0"], ["0"]]).set_color(GREEN)
        CCC2 = Matrix([["0"], ["0"], ["0"], ["1"], ["0"], ["0"]]).set_color(GREEN)
        AAA3 = Matrix([["0"], ["1"], ["0"], ["2"], ["0"], ["0"]]).set_color(RED)
        BBB3 = Matrix([["1"], ["0"], ["0"], ["0"], ["0"], ["0"]]).set_color(RED)
        CCC3 = Matrix([["0"], ["0"], ["0"], ["0"], ["1"], ["0"]]).set_color(RED)
        AAA4 = Matrix([["5"], ["0"], ["0"], ["0"], ["1"], ["0"]]).set_color(PINK)
        BBB4 = Matrix([["1"], ["0"], ["0"], ["0"], ["0"], ["0"]]).set_color(PINK)
        CCC4 = Matrix([["0"], ["0"], ["0"], ["0"], ["0"], ["1"]]).set_color(PINK)

        vec2 = [[0, 0, 0, 5]]
        vecc2 = Matrix(vec2).set_column_colors(BLUE, GREEN, RED, PINK).shift(3.5*DOWN + 2.3*LEFT)
        firs_a_vec_function = lambda x: vec2[0][3]*langrage_basis(x, 3)
        first_a_vec = lagrange_axes.plot(firs_a_vec_function, color = colors[3])
        a_vec = Group(AAA.copy(), AAA2.copy(), AAA3.copy(), AAA4.copy()).arrange().move_to(3.5*RIGHT)
        A1_lagr = MathTex("\\frac{5}{6}x^3 - 5x^2 + \\frac{55}{6}x - 5").shift(3.5*RIGHT).set_color(GREEN)

        constraint1 = Group(AAA, BBB, CCC).arrange().scale(0.7)
        constraint2 = Group(AAA2, BBB2, CCC2).arrange().scale(0.7)
        constraint3 = Group(AAA3, BBB3, CCC3).arrange().scale(0.7)
        constraint4 = Group(AAA4, BBB4, CCC4).arrange().scale(0.7)
        constraints_real = Group(constraint1, constraint2, constraint3, constraint4).arrange_in_grid().move_to(3*RIGHT)
        copies = Group(AAA.copy(), AAA2.copy(), AAA3.copy(), AAA4.copy())

        steps = [0,1,2,3]
        steps[0] = MathTex("\operatorname{int} = x*x").set_color(BLUE).move_to(constraints_real[0].get_center())
        steps[1] = MathTex("\operatorname{int}_2 = \operatorname{int}*x").set_color(GREEN).move_to(constraints_real[1].get_center())
        steps[2] = MathTex("\operatorname{int}_3 = \operatorname{int}_2 + x").set_color(RED).move_to(constraints_real[2].get_center())
        steps[3] = MathTex("\operatorname{out} = \operatorname{int}_3 + 5").set_color(PINK).move_to(constraints_real[3].get_center())
        steps_group = Group(steps[0], steps[1], steps[2], steps[3]).arrange(DOWN).move_to(3*RIGHT)
        equation = MathTex("x^{3} + x + 5 = 35", color=YELLOW)

        A2_lagr = MathTex("-\\frac{2}{3}x^3 + 5x^2 + -\\frac{34}{3}x + 8").scale(0.6)
        A3_lagr = MathTex("\\frac{1}{2}x^3 - 4x^2 + \\frac{19}{2}x - 6").scale(0.6)
        A4_lagr = MathTex("-x^3 + 7x^2 + -14x + 8").scale(0.6)
        A5_lagr = MathTex("\\frac{1}{6}x^3 - x^2 + \\frac{11}{6}x - 1").scale(0.6)
        A6_lagr = MathTex("0x^3 + 0x^2 + 0x + 0").scale(0.6)
        A = [A1_lagr, A2_lagr, A3_lagr, A4_lagr, A5_lagr, A6_lagr]
        # 6*LEFT+ 2*UP   2.5*LEFT + 2*UP
        for s in range(5):
            A[s+1].move_to(2.5*LEFT + 2*UP + 0.8*(s+1)*DOWN).set_color(GREEN)
        A_interp_succinct = [0,1,2,3,4,5]
        A_interp_succinct[0] = MathTex("A_{1}(x)").scale(0.7).move_to(A[1].get_center() + 0.8*UP).set_color(BLUE)
        A_interp_succinct[1] = MathTex("A_{2}(x)").scale(0.7).move_to(A[1].get_center()).set_color(BLUE)
        A_interp_succinct[2] = MathTex("A_{3}(x)").scale(0.7).move_to(A[2].get_center()).set_color(BLUE)
        A_interp_succinct[3] = MathTex("A_{4}(x)").scale(0.7).move_to(A[3].get_center()).set_color(BLUE)
        A_interp_succinct[4] = MathTex("A_{5}(x)").scale(0.7).move_to(A[4].get_center()).set_color(BLUE)
        A_interp_succinct[5] = MathTex("A_{6}(x)").scale(0.7).move_to(A[5].get_center()).set_color(BLUE)
        B_interp_succinct = [0,1,2,3,4,5]
        B_interp_succinct[0] = MathTex("B_{1}(x)").scale(0.7).set_color(BLUE)
        B_interp_succinct[1] = MathTex("B_{2}(x)").scale(0.7).set_color(BLUE)
        B_interp_succinct[2] = MathTex("B_{3}(x)").scale(0.7).set_color(BLUE)
        B_interp_succinct[3] = MathTex("B_{4}(x)").scale(0.7).set_color(BLUE)
        B_interp_succinct[4] = MathTex("B_{5}(x)").scale(0.7).set_color(BLUE)
        B_interp_succinct[5] = MathTex("B_{6}(x)").scale(0.7).set_color(BLUE)
        C_interp_succinct = [0,1,2,3,4,5]
        C_interp_succinct[0] = MathTex("C_{1}(x)").scale(0.7).set_color(BLUE)
        C_interp_succinct[1] = MathTex("C_{2}(x)").scale(0.7).set_color(BLUE)
        C_interp_succinct[2] = MathTex("C_{3}(x)").scale(0.7).set_color(BLUE)
        C_interp_succinct[3] = MathTex("C_{4}(x)").scale(0.7).set_color(BLUE)
        C_interp_succinct[4] = MathTex("C_{5}(x)").scale(0.7).set_color(BLUE)
        C_interp_succinct[5] = MathTex("C_{6}(x)").scale(0.7).set_color(BLUE)
        B_interp_succinct_group = Group(B_interp_succinct[0], B_interp_succinct[1], B_interp_succinct[2], B_interp_succinct[3], B_interp_succinct[4], B_interp_succinct[5]).scale(0.7).arrange(DOWN, buff=0.33).move_to(2*RIGHT)
        C_interp_succinct_group = Group(C_interp_succinct[0], C_interp_succinct[1], C_interp_succinct[2], C_interp_succinct[3], C_interp_succinct[4], C_interp_succinct[5]).scale(0.7).arrange(DOWN, buff=0.32).move_to(6.3*RIGHT)

        L_polys = [0,1,2,3]
        L_polys[0] = MathTex("L_{1}(x)").scale(0.7).move_to(a_vec[0].get_top() + 0.6*UP).set_color(ORANGE)
        L_polys[1] = MathTex("L_{2}(x)").scale(0.7).move_to(a_vec[1].get_top() + 0.6*UP).set_color(ORANGE)
        L_polys[2] = MathTex("L_{3}(x)").scale(0.7).move_to(a_vec[2].get_top() + 0.6*UP).set_color(ORANGE)
        L_polys[3] = MathTex("L_{4}(x)").scale(0.7).move_to(a_vec[3].get_top() + 0.6*UP).set_color(ORANGE)

        self.unique_polys(lagrange_axes, vecc, points, plots, interpoly_plot, interpoly_equ, interpo_point, vecc_plots)
        self.making_polys(lagrange_axes, vec2, vecc2, first_a_vec, plots, a_vec, A1_lagr, constraints_real, copies)
        self.return_to_vecs(a_vec, vecc2, A1_lagr, LIGHT, A_interp_succinct, A, L_polys)
        self.remind_where_vec_came_from(a_vec, constraints_real, steps_group, equation, A_interp_succinct, L_polys, LIGHT, B_interp_succinct_group, C_interp_succinct_group)

    def unique_polys(self, lagrange_axes, vecc, points, plots, interpoly_plot, interpoly_equ, interpo_point, vecc_plots):
        self.add(lagrange_axes)
        self.wait()
        self.play(FadeIn(vecc))
        self.wait()
        vecc_copy = vecc.copy()
        self.play(*[FadeTransform(vecc_copy[0][s], points[s]) for s in range(4)])
        self.wait()

        # show basis polys
        self.play(FadeIn(plots[0]))
        self.wait()
        self.play(FadeOut(plots[0]), FadeIn(plots[1]))
        self.wait()
        self.play(FadeTransform(plots[1], vecc_plots[0]))
        self.wait()
        self.play(FadeOut(vecc_plots[0]), FadeIn(plots[2]))
        self.wait()
        self.play(FadeTransform(plots[2], vecc_plots[1]))
        self.wait()
        self.play(FadeOut(vecc_plots[1]), FadeIn(plots[3]))
        self.wait()

        self.play(FadeIn(plots[0], vecc_plots[0], vecc_plots[1]))
        self.wait()
        self.play(FadeOut(plots[0], plots[3], vecc_plots[0], vecc_plots[1]), FadeIn(interpoly_plot))
        self.wait()
        self.play(Write(interpoly_equ))
        self.wait()
        self.play(FadeIn(interpo_point))
        self.wait()
        self.clear()
        self.wait()

    def making_polys(self, lagrange_axes, vec2, vecc2, first_a_vec, plots, a_vec, A1_lagr, constraints_real, copies):
        self.play(FadeIn(constraints_real))
        self.wait()
        self.add(copies)
        self.play(FadeOut(constraints_real))
        self.wait()

        # Make the grid of A vecs
        self.play(*[FadeTransform(copies[s], a_vec[s]) for s in range(4)])
        self.wait()
        self.play(*[FadeTransform(a_vec[s][0][0].copy(), vecc2[0][s]) for s in range(4)], FadeIn(vecc2[1], vecc2[2]), FadeOut(a_vec))
        self.wait()
        self.play(FadeIn(lagrange_axes))
        self.wait()
        vecc2_copy = vecc2.copy()
        colors = [BLUE, GREEN, RED, PINK]
        new_points = [Dot(lagrange_axes.coords_to_point(s+1,vec2[0][s]), color=colors[s]) for s in range(4)]
        self.play(*[FadeTransform(vecc2_copy[0][s], new_points[s]) for s in range(4)])
        self.wait()

        # show bases again
        self.play(FadeIn(plots[0]))
        self.wait()
        self.play(FadeOut(plots[0]), FadeIn(plots[1]))
        self.wait()
        self.play(FadeOut(plots[1]), FadeIn(plots[2]))
        self.wait()
        self.play(FadeOut(plots[2]), FadeIn(plots[3]))
        self.wait()
        basis_4 = MathTex("\\frac{1}{6}x^3 - x^2 + \\frac{11}{6}x - 1").shift(3.5*RIGHT).set_color(GREEN)
        self.play(Write(basis_4))
        self.wait()
        self.play(FadeTransform(plots[3], first_a_vec))
        self.wait()
        self.play(FadeTransform(basis_4, A1_lagr))
        self.wait()
        self.play(FadeOut(lagrange_axes, *[new_points[s] for s in range(4)], first_a_vec))
        self.wait()
        self.play(A1_lagr.animate.shift(6*LEFT+ 2*UP).scale(0.6))
        self.wait()

    def return_to_vecs(self, a_vec, vecc2, A1_lagr, LIGHT, A_interp_succinct, A, L_polys):
        a_vec_top_copies = [0,1,2,3]
        for s in range(4):
            a_vec_top_copies[s] = a_vec[s][0][0].copy()
        self.play(*[FadeTransform(vecc2[0][s], a_vec_top_copies[s]) for s in range(4)], FadeIn(a_vec), FadeOut(vecc2[1], vecc2[2]))
        self.wait()
        self.play(*[FadeOut(a_vec_top_copies[s]) for s in range(4)])


        Ai_values_rectangles = [0,1,2,3,4,5] 
        for s in range(5):
            Ai_values_rectangles[s] = Rectangle(width = 4.5, height = 0.5, color =LIGHT).move_to(3.5*RIGHT + 2*UP + 0.8*(s+1)*DOWN)
        self.play(*[FadeIn(Ai_values_rectangles[s]) for s in range(5)])
        self.wait()

        self.play(*[Write(L_polys[s]) for s in range(4)])
        self.wait()

        # present the interpolations in Lagrange basis
        interpolation_in_lagr_basis = [0,1,2,3,4]
        interpolation_in_lagr_basis[0] = MathTex("1*L_{1}(x)+1*L_{3}(x)").scale(0.7).move_to(A[1].get_center()).set_color(ORANGE)
        interpolation_in_lagr_basis[1] = MathTex("1*L_{2}(x)").scale(0.7).move_to(A[2].get_center()).set_color(ORANGE)
        interpolation_in_lagr_basis[2] = MathTex("2*L_{3}(x)").scale(0.7).move_to(A[3].get_center()).set_color(ORANGE)
        interpolation_in_lagr_basis[3] = MathTex("1*L_{4}(x)").scale(0.7).move_to(A[4].get_center()).set_color(ORANGE)
        interpolation_in_lagr_basis[4] = MathTex("0").scale(0.7).move_to(A[5].get_center()).set_color(ORANGE)

        self.play(*[FadeTransform(Ai_values_rectangles[s], interpolation_in_lagr_basis[s]) for s in range(5)])
        self.wait()
        
        # introduce the interpolated A equations in x basis
        self.play(*[FadeTransform(interpolation_in_lagr_basis[s], A[s+1]) for s in range(5)])
        self.wait()


        self.play(*[FadeTransform(A[s], A_interp_succinct[s]) for s in range(6)])
        self.wait()

        # plane = NumberPlane()
        # self.add(plane)
        self.wait()
        self.play(*[FadeOut(L_polys[s]) for s in range(4)])
        self.wait()


    def remind_where_vec_came_from(self, a_vec, constraints_real, steps_group, equation, A_interp_succinct, L_polys, LIGHT, B_interp_succinct_group, C_interp_succinct_group):
        constraints_real_first_copy = [0,1,2,3]
        for s in range(4):
            constraints_real_first_copy[s] = constraints_real[s][0].copy()
        self.play(*[FadeTransform(a_vec[s], constraints_real_first_copy[s]) for s in range(4)], FadeIn(constraints_real))
        # self.play(FadeIn(constraints_real))
        self.play(*[FadeOut(constraints_real_first_copy[s]) for s in range(4)])
        self.wait()

        # link vector triples to constraints
        surrounder = [0,1,2,3]
        surrounder[0] = SurroundingRectangle(constraints_real[0], color = BLUE)
        surrounder[1] = SurroundingRectangle(constraints_real[1], color = GREEN)
        surrounder[2] = SurroundingRectangle(constraints_real[2], color = RED)
        surrounder[3] = SurroundingRectangle(constraints_real[3], color = PINK)
        self.play(*[FadeTransform(surrounder[s], steps_group[s]) for s in range(4)], *[FadeOut(constraints_real[s]) for s in range(4)])
        self.wait()

        # link to original equation
        final_surrounder = SurroundingRectangle(steps_group, color = YELLOW)
        self.play(FadeTransform(final_surrounder, equation.move_to(steps_group.get_center() + 3*UP)))
        self.wait()

        # create constraints from equations, and turn those constraints into vector triples
        self.play(FadeTransform(equation, final_surrounder))
        self.play(FadeOut(final_surrounder))
        self.wait()
        self.play(*[FadeTransform(steps_group[s], surrounder[s]) for s in range(4)], *[FadeIn(constraints_real[s]) for s in range(4)])
        self.play(*[FadeOut(surrounder[s]) for s in range(4)])
        self.wait()

        a_vecs = Group(constraints_real[0][0].copy(), constraints_real[1][0].copy(), constraints_real[2][0].copy(), constraints_real[3][0].copy()).arrange(buff = 0.1)
        b_vecs = Group(constraints_real[0][1].copy(), constraints_real[1][1].copy(), constraints_real[2][1].copy(), constraints_real[3][1].copy()).arrange(buff = 0.1)
        c_vecs = Group(constraints_real[0][2].copy(), constraints_real[1][2].copy(), constraints_real[2][2].copy(), constraints_real[3][2].copy()).arrange(buff = 0.1)
        vec_groups = Group(a_vecs, b_vecs, c_vecs).arrange(buff = 1.3)
        A_interp_succinct_group = Group(A_interp_succinct[0], A_interp_succinct[1], A_interp_succinct[2], A_interp_succinct[3], A_interp_succinct[4], A_interp_succinct[5])
        self.play(A_interp_succinct_group.animate.scale(0.7).move_to(2.4*LEFT))
        self.play(*[FadeTransform(constraints_real[s][0], a_vecs[s]) for s in range(4)], *[FadeTransform(constraints_real[s][1], b_vecs[s]) for s in range(4)], *[FadeTransform(constraints_real[s][2], c_vecs[s]) for s in range(4)])
        self.wait()

        L_basis_group = Group(L_polys[0], L_polys[1], L_polys[2], L_polys[3]).scale(0.7).arrange(buff = 0.18).move_to(a_vecs.get_center() + 2*UP)
        self.play(*[Write(L_basis_group[s]) for s in range(4)])
        self.wait()

        reccys = [0,1,2,3,4,5]
        for s in range(6):
            reccys[s] = Rectangle(width = 2.8, height = 0.4, color = LIGHT).move_to(a_vecs.get_center() + 1.4*UP + s*0.56*DOWN)
        reccys2 = [0,1,2,3,4,5]
        for s in range(6):
            reccys2[s] = Rectangle(width = 2.8, height = 0.4, color = LIGHT).move_to(b_vecs.get_center() + 1.4*UP + s*0.56*DOWN)
        reccys3 = [0,1,2,3,4,5]
        for s in range(6):
            reccys3[s] = Rectangle(width = 2.8, height = 0.4, color = LIGHT).move_to(c_vecs.get_center() + 1.4*UP + s*0.56*DOWN)

        self.play(*[FadeIn(reccys[s]) for s in range(6)])
        self.wait()

        self.play(*[FadeTransform(reccys[s], A_interp_succinct_group[s]) for s in range(6)])
        self.wait()

        # Display Lagrange bases above the b and c vec groups
        L_basis_group2 = L_basis_group.copy().move_to(b_vecs.get_center() + 2*UP)
        L_basis_group3 = L_basis_group.copy().move_to(c_vecs.get_center() + 2*UP)
        self.play(*[Write(L_basis_group2[s]) for s in range(4)], *[Write(L_basis_group3[s]) for s in range(4)])
        self.wait()

        # Rectangles around b_vec entries
        self.play(*[FadeIn(reccys2[s]) for s in range(6)])
        self.wait()
        self.play(*[ReplacementTransform(reccys2[s], B_interp_succinct_group[s]) for s in range(6)])
        self.wait()

        # Rectangles around c_vec entries
        self.play(*[FadeIn(reccys3[s]) for s in range(6)])
        self.wait()
        self.play(*[ReplacementTransform(reccys3[s], C_interp_succinct_group[s]) for s in range(6)])
        self.wait()

        # Fade out constraint vectors and Lagrange bases
        self.play(FadeOut(L_basis_group, L_basis_group2, L_basis_group3, a_vecs, b_vecs, c_vecs))
        self.wait()

        # Make lagr interp into actual vectors rather than just grouping, and bring in sol vec again
        A_lagr_vec = Matrix([["A_1(x)"], ["A_2(x)"], ["A_3(x)"], ["A_4(x)"], ["A_5(x)"], ["A_6(x)"]]).set_color(BLUE)
        B_lagr_vec = Matrix([["B_1(x)"], ["B_2(x)"], ["B_3(x)"], ["B_4(x)"], ["B_5(x)"], ["B_6(x)"]]).set_color(BLUE)
        C_lagr_vec = Matrix([["C_1(x)"], ["C_2(x)"], ["C_3(x)"], ["C_4(x)"], ["C_5(x)"], ["C_6(x)"]]).set_color(BLUE)
        sol_vec = Matrix([[1], ["x"], ["\operatorname{int}"], ["\operatorname{int}_2"], ["\operatorname{int}_3"], ["\operatorname{out}"]]).set_color(YELLOW)
        times = Tex("*")
        equals = Tex("=")
        minus = Tex("-")
        zero = Tex("0")
        combined_vectors = Group(sol_vec, A_lagr_vec, times, sol_vec.copy(), B_lagr_vec, equals, sol_vec.copy(), C_lagr_vec).arrange()
        combined_vectors_rearranged = Group(sol_vec.copy(), A_lagr_vec.copy(), times.copy(), sol_vec.copy(), B_lagr_vec.copy(), minus.copy(), sol_vec.copy(), C_lagr_vec.copy(), equals.copy(), zero).arrange()
        self.play(*[FadeTransform(A_interp_succinct_group[s], A_lagr_vec[0][s]) for s in range(6)], FadeIn(A_lagr_vec[1], A_lagr_vec[2]))
        self.play(*[FadeTransform(B_interp_succinct_group[s], B_lagr_vec[0][s]) for s in range(6)], FadeIn(B_lagr_vec[1], B_lagr_vec[2]))
        self.play(*[FadeTransform(C_interp_succinct_group[s], C_lagr_vec[0][s]) for s in range(6)], FadeIn(C_lagr_vec[1], C_lagr_vec[2]))
        self.wait()
        self.play(FadeIn(combined_vectors[0], combined_vectors[2], combined_vectors[3], combined_vectors[5], combined_vectors[6]))
        self.wait()
        self.play(*[FadeTransform(combined_vectors[s], combined_vectors_rearranged[s]) for s in range(8)], FadeIn(combined_vectors_rearranged[8], combined_vectors_rearranged[9]))
        self.wait()

        # evaluating the poly returns us to one of the constraints
        A_lagr_vec_at_zero = Matrix([["A_1(0)"], ["A_2(0)"], ["A_3(0)"], ["A_4(0)"], ["A_5(0)"], ["A_6(0)"]]).set_color(BLUE).move_to(A_lagr_vec.get_center())
        B_lagr_vec_at_zero = Matrix([["B_1(0)"], ["B_2(0)"], ["B_3(0)"], ["B_4(0)"], ["B_5(0)"], ["B_6(0)"]]).set_color(BLUE).move_to(B_lagr_vec.get_center())
        C_lagr_vec_at_zero = Matrix([["C_1(0)"], ["C_2(0)"], ["C_3(0)"], ["C_4(0)"], ["C_5(0)"], ["C_6(0)"]]).set_color(BLUE).move_to(C_lagr_vec.get_center())
        grouped_at_zero = Group(sol_vec.copy(), A_lagr_vec_at_zero.copy(), times.copy(), sol_vec.copy(), B_lagr_vec_at_zero.copy(), minus.copy(), sol_vec.copy(), C_lagr_vec_at_zero.copy(), equals.copy(), zero.copy()).arrange()
        A_lagr_vec_at_zero_value = Matrix([["0"], ["1"], ["0"], ["0"], ["0"], ["0"]]).set_color(BLUE).move_to(A_lagr_vec.get_center())
        B_lagr_vec_at_zero_value = Matrix([["0"], ["1"], ["0"], ["0"], ["0"], ["0"]]).set_color(BLUE).move_to(B_lagr_vec.get_center())
        C_lagr_vec_at_zero_value = Matrix([["0"], ["0"], ["1"], ["0"], ["0"], ["0"]]).set_color(BLUE).move_to(C_lagr_vec.get_center())
        grouped_at_zero_value = Group(sol_vec.copy(), A_lagr_vec_at_zero_value.copy(), times.copy(), sol_vec.copy(), B_lagr_vec_at_zero_value.copy(), minus.copy(), sol_vec.copy(), C_lagr_vec_at_zero_value.copy(), equals.copy(), zero.copy()).arrange()
        self.play(*[FadeTransform(combined_vectors_rearranged[s], grouped_at_zero[s]) for s in range(10)])
        self.wait()
        self.play(*[FadeTransform(grouped_at_zero[s], grouped_at_zero_value[s]) for s in range(10)])
        self.wait()
        constraint1 = MathTex("\\operatorname{int} = x*x").set_color(BLUE).move_to(3.5*DOWN)
        self.play(FadeIn(constraint1))
        self.wait()
        self.play(*[FadeTransform(grouped_at_zero_value[s], combined_vectors_rearranged[s]) for s in range(10)], FadeOut(constraint1))
        self.wait()