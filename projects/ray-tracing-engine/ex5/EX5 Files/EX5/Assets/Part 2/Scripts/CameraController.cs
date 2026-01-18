using UnityEngine;
using UnityEngine.InputSystem;

public class LookAround : MonoBehaviour
{
    [Tooltip("Controls how fast the camera rotates.")]
    public float sensitivity = 2.0f;

    private float rotationX;
    private float rotationY;

    private InputAction lookAction;
    private InputAction clickAction;

    private void Awake()
    {
        lookAction = new InputAction("look", binding: "<Mouse>/delta");
        clickAction = new InputAction("click", binding: "<Mouse>/leftButton");
    }

    private void OnEnable()
    {
        lookAction.Enable();
        clickAction.Enable();
    }

    private void OnDisable()
    {
        lookAction.Disable();
        clickAction.Disable();
    }
    
    private void Start()
    {
        var startAngles = transform.eulerAngles;
        rotationY = startAngles.y;
        rotationX = startAngles.x;
    }

    private void Update()
    {
        if (!clickAction.IsPressed()) { return; }
        
        var mouseDelta = lookAction.ReadValue<Vector2>();

        var mouseX = mouseDelta.x * sensitivity * 0.1f;
        var mouseY = mouseDelta.y * sensitivity * 0.1f;

        rotationY += mouseX;
        rotationX -= mouseY;

        // Clamp the vertical rotation (pitch) to prevent the camera from flipping upside down.
        // A range of -90 to 90 degrees is standard.
        rotationX = Mathf.Clamp(rotationX, -90f, 90f);
        transform.localRotation = Quaternion.Euler(rotationX, rotationY, 0);
    }
}